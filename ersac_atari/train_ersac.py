
from dataclasses import dataclass
from typing import Literal
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import AtariEncoder, QEnsemble, DiscretePolicy, EpinetQ
from .uncertainty import worst_case_box, worst_case_ellipsoid


@dataclass
class TrainCfg:
    algo: Literal['sac_n', 'ersac_epi'] = 'ersac_epi'
    n_ensemble: int = 10
    z_dim: int = 16
    ellipsoid_radius: float = 3.0
    alpha: float = 0.01
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    target_tau: float = 0.005
    device: str = 'cuda'
    grad_clip: float = 10.0
    max_steps: int = 500_000
    log_interval: int = 1000
    eval_interval: int = 25_000


class TargetModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = copy.deepcopy(module)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def soft_update(self, src: nn.Module, tau: float):
        for p_t, p in zip(self.module.parameters(), src.parameters()):
            p_t.data.lerp_(p.data, tau)


class SACNAgent(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, cfg: TrainCfg):
        super().__init__()
        self.encoder = AtariEncoder(obs_channels)
        self.policy = DiscretePolicy(self.encoder.out_dim, n_actions)
        self.qens = QEnsemble(self.encoder, self.encoder.out_dim, n_actions, cfg.n_ensemble)
        self.qens_tgt = TargetModule(self.qens)
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.n_actions = n_actions

    def pi_logits(self, obs):
        feat = self.encoder(obs)
        return self.policy(feat)


class ERSACEpiAgent(nn.Module):
    def __init__(self, obs_channels: int, n_actions: int, cfg: TrainCfg):
        super().__init__()
        self.encoder = AtariEncoder(obs_channels)
        self.policy = DiscretePolicy(self.encoder.out_dim, n_actions)
        self.qepi = EpinetQ(self.encoder, self.encoder.out_dim, n_actions, z_dim=cfg.z_dim)
        self.qepi_tgt = TargetModule(self.qepi)
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma
        self.n_actions = n_actions

    def pi_logits(self, obs):
        feat = self.encoder(obs)
        return self.policy(feat)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    pi = torch.softmax(logits, dim=-1)
    logpi = torch.log_softmax(logits, dim=-1)
    return -(pi * logpi).sum(dim=-1)


def policy_expectation(pi_logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pi = torch.softmax(pi_logits, dim=-1)
    return (pi * q).sum(dim=-1)


def sacn_step(agent: SACNAgent,
              opt_pi: optim.Optimizer,
              opt_q: optim.Optimizer,
              batch,
              cfg: TrainCfg,
              global_step: int):
    device = cfg.device
    obs = batch.obs.to(device, non_blocking=True)
    act = batch.act.to(device)
    rew = batch.rews.to(device)
    nxt = batch.next_obs.to(device, non_blocking=True)
    done = batch.done.to(device)
    trunc = batch.trunc.to(device)
    not_done = 1.0 - torch.clamp(done + trunc, 0, 1)

    # Critic update
    opt_q.zero_grad(set_to_none=True)
    with torch.no_grad():
        nxt_logits = agent.pi_logits(nxt)
        q_list_tgt = agent.qens_tgt.module.forward_all(nxt)
        q_min_tgt = worst_case_box(q_list_tgt, nxt_logits)
        exp_q = policy_expectation(nxt_logits, q_min_tgt)
        ent = entropy_from_logits(nxt_logits)
        y = rew + cfg.gamma * not_done * (exp_q - cfg.alpha * ent)

    q_list = agent.qens.forward_all(obs)
    q_pred = torch.stack([qi.gather(1, act.view(-1,1)).squeeze(1) for qi in q_list], dim=0)
    q_loss = ((q_pred - y.unsqueeze(0)) ** 2).mean()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.qens.parameters(), cfg.grad_clip)
    opt_q.step()

    # Policy update
    opt_pi.zero_grad(set_to_none=True)
    pi_logits = agent.pi_logits(obs)
    with torch.no_grad():
        q_min_now = agent.qens.forward_min(obs)
    j_pi = (policy_expectation(pi_logits, q_min_now) - cfg.alpha * entropy_from_logits(pi_logits)).mean()
    pi_loss = -j_pi
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.grad_clip)
    opt_pi.step()

    # Target update
    agent.qens_tgt.soft_update(agent.qens, cfg.target_tau)

    if (global_step % cfg.log_interval) == 0:
        return {"q_loss": float(q_loss.item()), "pi_loss": float(pi_loss.item())}
    return None


def ersac_epi_step(agent: ERSACEpiAgent,
                   opt_pi: optim.Optimizer,
                   opt_q: optim.Optimizer,
                   batch,
                   cfg: TrainCfg,
                   global_step: int):
    device = cfg.device
    obs = batch.obs.to(device, non_blocking=True)
    act = batch.act.to(device)
    rew = batch.rews.to(device)
    nxt = batch.next_obs.to(device, non_blocking=True)
    done = batch.done.to(device)
    trunc = batch.trunc.to(device)
    not_done = 1.0 - torch.clamp(done + trunc, 0, 1)

    # Critic update (Epinet)
    opt_q.zero_grad(set_to_none=True)
    with torch.no_grad():
        nxt_logits = agent.pi_logits(nxt)
        tgt_out = agent.qepi_tgt.module(nxt)
        radius = torch.full((nxt.shape[0],), cfg.ellipsoid_radius, device=device)
        q_star_tgt = worst_case_ellipsoid(tgt_out.mu, tgt_out.Sigma, nxt_logits, radius)
        exp_q = policy_expectation(nxt_logits, q_star_tgt)
        ent = entropy_from_logits(nxt_logits)
        y = rew + cfg.gamma * not_done * (exp_q - cfg.alpha * ent)

    out = agent.qepi(obs)
    q_pred = out.mu.gather(1, act.view(-1,1)).squeeze(1)
    q_loss = ((q_pred - y) ** 2).mean()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.qepi.parameters(), cfg.grad_clip)
    opt_q.step()

    # Policy update with worst-case ellipsoid
    opt_pi.zero_grad(set_to_none=True)
    pi_logits = agent.pi_logits(obs)
    with torch.no_grad():
        out_now = agent.qepi(obs)
        radius_now = torch.full((obs.shape[0],), cfg.ellipsoid_radius, device=device)
        q_star_now = worst_case_ellipsoid(out_now.mu, out_now.Sigma, pi_logits, radius_now)
    j_pi = (policy_expectation(pi_logits, q_star_now) - cfg.alpha * entropy_from_logits(pi_logits)).mean()
    pi_loss = -j_pi
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), cfg.grad_clip)
    opt_pi.step()

    # Target update
    agent.qepi_tgt.soft_update(agent.qepi, cfg.target_tau)

    if (global_step % cfg.log_interval) == 0:
        return {"q_loss": float(q_loss.item()), "pi_loss": float(pi_loss.item())}
    return None
