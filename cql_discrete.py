# cql_discrete.py
"""
Conservative Q-Learning (CQL) for discrete action spaces
Benchmark implementation for comparison with ERSAC methods

Based on: "Conservative Q-Learning for Offline Reinforcement Learning"
Kumar et al., NeurIPS 2020
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn

from common_offline_discrete import AlgoConfig, DiscretePolicy, DiscreteQ, polyak_update, clamp_like


class CQLAgent:
    """CQL agent for discrete action spaces with image observations"""
    
    def __init__(self, encoder: nn.Module, obs_shape: Tuple[int, int, int], num_actions: int,
                 cfg: AlgoConfig, cql_alpha: float = 1.0, actor_hidden: int = 256, critic_hidden: int = 256):
        """
        Args:
            encoder: CNN encoder for image observations
            obs_shape: (C, H, W) shape of observations
            num_actions: Number of discrete actions
            cfg: Algorithm configuration
            cql_alpha: CQL penalty weight (higher = more conservative)
            actor_hidden: Hidden size for policy network
            critic_hidden: Hidden size for Q-network
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_actions = num_actions

        # Shared encoder (conv for pixels)
        self.encoder = encoder.to(self.device)

        # Policy and Q-networks
        embed_dim = getattr(encoder, 'embed_dim', 512)
        self.policy = DiscretePolicy(embed_dim, num_actions, hidden=actor_hidden).to(self.device)
        self.q = DiscreteQ(embed_dim, num_actions, hidden=critic_hidden).to(self.device)
        self.q_tgt = DiscreteQ(embed_dim, num_actions, hidden=critic_hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        # Entropy temperature for actor
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -float(num_actions) if cfg.target_entropy is None else cfg.target_entropy

        # Optimizers
        # Train encoder with BOTH actor and critic for better representation learning
        self.opt_critic = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q.parameters()), 
            lr=cfg.lr_critic
        )
        self.opt_actor = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        # Hyperparameters
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.cql_alpha = cql_alpha

    @property
    def alpha(self):
        """Current entropy temperature"""
        return self.log_alpha.exp()

    def update(self, batch):
        """
        Update agent from a batch of offline data
        
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

        # ------ Encode states ------
        z = self.encoder(S)
        z2 = self.encoder(S2).detach()

        # ------ Critic update ------
        with torch.no_grad():
            # Soft value target using current policy
            pi2, log_pi2, _ = self.policy(z2)
            q2_tgt = self.q_tgt(z2)  # (B, A)
            soft_v2 = (pi2 * (q2_tgt - self.alpha * log_pi2)).sum(-1)
            soft_v2 = clamp_like(soft_v2, self.cfg.clamp_value_min, self.cfg.clamp_value_max)
            y = R + (1.0 - D) * self.gamma * soft_v2
            y = clamp_like(y, self.cfg.clamp_value_min, self.cfg.clamp_value_max)

        # Q-values and TD loss
        q_pred = self.q(z)  # (B, A)
        q_on_a = q_pred.gather(1, A.unsqueeze(1)).squeeze(1)  # (B,)
        td_loss = F.smooth_l1_loss(q_on_a, y)

        # ------ CQL penalty: logsumexp(Q) - Q(s,a_data) ------
        # This encourages Q-values to be lower for unseen actions
        logsumexp = torch.logsumexp(q_pred, dim=-1).mean()
        data_q = q_on_a.mean()
        cql_pen = self.cql_alpha * (logsumexp - data_q)

        critic_loss = td_loss + cql_pen

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.q.parameters()), 
            self.cfg.grad_norm_clip
        )
        self.opt_critic.step()

        # ------ Actor update ------
        # Re-encode with updated encoder
        z_actor = self.encoder(S).detach()  # Detach - encoder already trained by critic
        pi, log_pi, _ = self.policy(z_actor)
        q_for_pi = self.q(z_actor).detach()  # Don't update Q through actor loss
        
        # Standard SAC actor objective: E[Q(s,a) - α*log π(a|s)]
        actor_obj = (pi * (q_for_pi - self.alpha * log_pi)).sum(-1).mean()
        actor_loss = -actor_obj

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
        self.opt_actor.step()

        # ------ Temperature update ------
        with torch.no_grad():
            pi_det, log_pi_det, _ = self.policy(self.encoder(S))
            # Entropy error weighted by policy
            ent_err = (-(log_pi_det + self.target_entropy) * pi_det).sum(-1)  # (B,)
        
        alpha_loss = (self.alpha * ent_err).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        # ------ Target network update ------
        polyak_update(self.q_tgt, self.q, self.tau)

        # Compute additional metrics for debugging
        with torch.no_grad():
            mean_q = q_pred.mean().item()
            max_q = q_pred.max(dim=-1)[0].mean().item()
            q_data = q_on_a.mean().item()
            # Action agreement: how often does greedy policy match data?
            greedy_actions = q_pred.argmax(dim=-1)
            action_agreement = (greedy_actions == A).float().mean().item()

        return {
            "critic_loss": float(critic_loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_pen": float(cql_pen.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_q": mean_q,
            "max_q": max_q,
            "q_data": q_data,
            "action_agree": action_agreement,
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
