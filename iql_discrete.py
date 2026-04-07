# iql_discrete.py
"""
Implicit Q-Learning (IQL) for discrete action spaces
Benchmark implementation for comparison with ERSAC methods

Based on: "Offline Reinforcement Learning with Implicit Q-Learning"
Kostrikov et al., ICLR 2022

IQL avoids explicit policy constraints (like CQL's penalty) by learning
a value function via expectile regression, then extracting policy via
advantage-weighted regression (AWR).
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

from common_offline_discrete import (
    AlgoConfig, DiscretePolicy, DiscreteQ, ValueNet,
    expectile_loss, polyak_update, clamp_like
)


class IQLAgent:
    """IQL agent for discrete action spaces with image observations"""
    
    def __init__(self, encoder: nn.Module, obs_shape: Tuple[int, int, int], num_actions: int,
                 cfg: AlgoConfig, expectile_tau: float = 0.7, beta_adv: float = 3.0,
                 actor_hidden: int = 256, critic_hidden: int = 256, value_hidden: int = 256):
        """
        Args:
            encoder: CNN encoder for image observations
            obs_shape: (C, H, W) shape of observations
            num_actions: Number of discrete actions
            cfg: Algorithm configuration
            expectile_tau: Expectile level for value learning (0.7 = 70th percentile)
            beta_adv: Temperature for advantage weighting in policy update
            actor_hidden: Hidden size for policy network
            critic_hidden: Hidden size for Q-network
            value_hidden: Hidden size for value network
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_actions = num_actions

        # Shared encoder
        self.encoder = encoder.to(self.device)
        embed_dim = getattr(encoder, 'embed_dim', 512)

        # Networks: policy, Q, and value
        self.policy = DiscretePolicy(embed_dim, num_actions, hidden=actor_hidden).to(self.device)
        self.q = DiscreteQ(embed_dim, num_actions, hidden=critic_hidden).to(self.device)
        self.q_tgt = DiscreteQ(embed_dim, num_actions, hidden=critic_hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.v = ValueNet(embed_dim, hidden=value_hidden).to(self.device)
        self.v_tgt = ValueNet(embed_dim, hidden=value_hidden).to(self.device)
        self.v_tgt.load_state_dict(self.v.state_dict())

        # Optimizers
        # Train encoder primarily with Q-network (like CQL)
        self.opt_critic = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q.parameters()),
            lr=cfg.lr_critic
        )
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=cfg.lr_value)
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)

        # Hyperparameters
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.expectile_tau = expectile_tau
        self.beta_adv = beta_adv

    def update(self, batch):
        """
        Update agent from a batch of offline data
        
        IQL update has 3 stages:
        1. Update V via expectile regression on Q - V
        2. Update Q via standard TD learning to V(s') + regularization for discrete actions
        3. Update policy via advantage-weighted behavioral cloning
        
        Args:
            batch: (S, A, R, S2, D) tuple of tensors
                S: (B, C, H, W) states
                A: (B,) actions
                R: (B,) rewards
                S2: (B, C, H, W) next states
                D: (B,) done flags
                
        Returns:
            Dict of training metrics
        """
        S, A, R, S2, D = [b.to(self.device) for b in batch]
        A = A.long()
        D = D.float()

        # Encode states (will be updated via critic optimizer)
        z = self.encoder(S)
        z2 = self.encoder(S2).detach()

        # ----- 1) Update V by expectile regression on (Q - V) -----
        # V learns an expectile of Q(s, a_data), which provides implicit conservatism
        with torch.no_grad():
            # Use TARGET Q for stability in V update
            q_tgt_all = self.q_tgt(z.detach())  # (B, A)
            q_tgt_on_a = q_tgt_all.gather(1, A.unsqueeze(1)).squeeze(1)  # (B,)
        
        v_pred = self.v(z.detach())  # Detach - encoder trained by critic
        
        # Expectile loss: emphasizes Q-values above V (when tau > 0.5)
        # This makes V track a high quantile of Q, providing conservatism
        v_loss = expectile_loss(q_tgt_on_a - v_pred, tau=self.expectile_tau)
        
        self.opt_v.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), self.cfg.grad_norm_clip)
        self.opt_v.step()

        # ----- 2) Update Q (and encoder) with TD to r + gamma * V_target(s') -----
        # For discrete IQL, we need Q to differentiate between actions.
        # Standard approach: train Q on data actions, but add a regularization
        # to prevent all Q-values from collapsing to the same value.
        with torch.no_grad():
            v_next = self.v_tgt(z2)  # Use TARGET V for stability
            y = R + (1.0 - D) * self.gamma * v_next
            y = clamp_like(y, self.cfg.clamp_value_min, self.cfg.clamp_value_max)
        
        q_pred = self.q(z)  # (B, A) - Gradient flows through encoder
        q_on_a = q_pred.gather(1, A.unsqueeze(1)).squeeze(1)  # (B,)
        
        # TD loss for data actions
        td_loss = F.smooth_l1_loss(q_on_a, y)
        
        # Regularization for discrete IQL: encourage Q(s, a_data) > Q(s, a_other)
        # This is similar to CQL but softer - push data actions to be preferred
        # logsumexp(Q) - Q(a_data) should be minimized
        logsumexp_q = torch.logsumexp(q_pred, dim=-1)  # (B,)
        q_reg = 0.5 * (logsumexp_q - q_on_a).mean()  # Soft CQL-like regularization
        
        q_loss = td_loss + q_reg

        self.opt_critic.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q.parameters()) + list(self.encoder.parameters()),
            self.cfg.grad_norm_clip
        )
        self.opt_critic.step()

        # Polyak update for target networks (both Q and V)
        polyak_update(self.q_tgt, self.q, self.tau)
        polyak_update(self.v_tgt, self.v, self.tau)

        # ----- 3) Advantage-weighted policy update -----
        # Policy is trained via AWR: maximize log π(a|s) weighted by exp(β * A(s,a))
        # where A(s,a) = Q(s,a) - V(s)
        z_actor = self.encoder(S).detach()  # Detach - encoder trained by critic
        pi, log_pi, _ = self.policy(z_actor)
        
        with torch.no_grad():
            q_for_adv = self.q(z_actor)  # (B, A)
            v_for_adv = self.v(z_actor)  # (B,)
            adv = q_for_adv - v_for_adv.unsqueeze(-1)  # (B, A)
            
            # Advantage weights: exp(β * advantage)
            # Clamp advantages for numerical stability
            adv_clamped = torch.clamp(adv, -100.0 / self.beta_adv, 100.0 / self.beta_adv)
            weights = torch.exp(self.beta_adv * adv_clamped)  # (B, A)
            
            # Normalize weights per batch (optional but helps stability)
            weights = weights / weights.mean()
        
        # Weighted negative log-likelihood on data actions
        log_pi_on_a = log_pi.gather(1, A.unsqueeze(1)).squeeze(1)  # (B,)
        weights_on_a = weights.gather(1, A.unsqueeze(1)).squeeze(1)  # (B,)
        
        actor_loss = -(weights_on_a * log_pi_on_a).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
        self.opt_actor.step()

        # Compute additional metrics for debugging (like CQL)
        with torch.no_grad():
            mean_q = q_for_adv.mean().item()
            q_data = q_for_adv.gather(1, A.unsqueeze(1)).squeeze(1).mean().item()
            mean_v = v_for_adv.mean().item()
            # Action agreement: how often does greedy policy match data?
            greedy_actions = q_for_adv.argmax(dim=-1)
            action_agreement = (greedy_actions == A).float().mean().item()
            # Advantage for data actions
            adv_data = adv.gather(1, A.unsqueeze(1)).squeeze(1).mean().item()

        return {
            "critic_loss": float(q_loss.item()),
            "td_loss": float(td_loss.item()),
            "q_reg": float(q_reg.item()),
            "value_loss": float(v_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mean_q": mean_q,
            "q_data": q_data,
            "mean_v": mean_v,
            "adv_data": adv_data,
            "action_agree": action_agreement,
            "mean_weight": float(weights_on_a.mean().item()),
        }

    @torch.no_grad()
    def act(self, obs: torch.Tensor, eval_mode: bool = False):
        """
        Select action for given observation
        
        Args:
            obs: (C, H, W) or (B, C, H, W) observation
            eval_mode: If True, use greedy action; otherwise sample
            
        Returns:
            action: Selected action(s)
        """
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        
        obs = obs.to(self.device)
        z = self.encoder(obs)
        pi, log_pi, _ = self.policy(z)
        
        if eval_mode:
            # Greedy action
            action = pi.argmax(-1)
        else:
            # Sample from policy
            action = torch.multinomial(pi, num_samples=1).squeeze(-1)
        
        return action.cpu().numpy() if action.numel() > 1 else action.item()
