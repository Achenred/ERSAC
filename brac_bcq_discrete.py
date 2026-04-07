# brac_bcq_discrete.py
"""
BRAC + BCQ for discrete action spaces
Combines Behavior Regularization (BRAC) with Batch-Constrained Q-learning (BCQ)

BRAC: Behavior Regularized Actor Critic (Wu et al., ICML 2019)
- Adds KL(π||μ) penalty to keep policy close to behavior
- λ_KL controls how much to constrain to behavioral policy

BCQ: Batch-Constrained Q-learning (Fujimoto et al., NeurIPS 2019)
- Filters out unlikely actions based on behavior policy threshold
- Only considers actions with μ(a|s) > threshold

This combines both approaches:
1. Learn behavior policy μ from data
2. Use BCQ filtering to restrict action space
3. Add BRAC KL penalty for additional conservatism
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_offline_discrete import AlgoConfig, DiscretePolicy, DiscreteQ, CNNEncoder, polyak_update, clamp_like


def gather_qa(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """
    Gather Q-values for specific actions
    
    Args:
        q_values: (B, A) Q-values for all actions
        actions: (B,) action indices
        
    Returns:
        (B,) Q-values for selected actions
    """
    return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


def kl_pi_mu(log_pi: torch.Tensor, log_mu: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence KL(π||μ) for categorical distributions
    
    KL(π||μ) = Σ_a π(a|s) * (log π(a|s) - log μ(a|s))
    
    Args:
        log_pi: (B, A) log probabilities of policy π
        log_mu: (B, A) log probabilities of behavior μ
        
    Returns:
        (B,) KL divergence per state
    """
    pi = log_pi.exp()
    return (pi * (log_pi - log_mu)).sum(-1)


def mask_logits_with_mu(logits: torch.Tensor, mu_prob: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    BCQ-style action filtering: mask out actions with μ(a|s) < threshold
    
    Args:
        logits: (B, A) raw logits before softmax
        mu_prob: (B, A) behavior policy probabilities
        threshold: minimum probability threshold
        
    Returns:
        (B, A) masked logits with -inf for filtered actions
    """
    mask = mu_prob < threshold
    masked_logits = logits.clone()
    masked_logits[mask] = -1e10  # Large negative number → exp ≈ 0
    return masked_logits


class BRACAgentDiscrete(nn.Module):
    """
    BRAC + BCQ agent for discrete action spaces with image observations
    
    Components:
    - μ (behavior policy): Learns to imitate dataset actions
    - π (target policy): Optimized with BRAC/BCQ constraints
    - Q (critic): Standard Q-learning
    """
    
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int, embed_dim: int = 512,
                 hidden: int = 256, cfg: AlgoConfig = None, lambda_kl: float = 1.0,
                 entropy_coef: float = 0.0, bcq_threshold: Optional[float] = None):
        """
        Args:
            obs_shape: (C, H, W) observation shape
            num_actions: Number of discrete actions
            embed_dim: Encoder output dimension
            hidden: Hidden layer size for policy/Q networks
            cfg: Algorithm configuration
            lambda_kl: Weight for KL(π||μ) penalty (BRAC)
            entropy_coef: Weight for entropy bonus (optional)
            bcq_threshold: If set, filter actions with μ(a|s) < threshold (BCQ)
                          Typical values: 0.1, 0.3 (None = no filtering)
        """
        super().__init__()
        self.cfg = cfg if cfg is not None else AlgoConfig()
        self.device = torch.device(self.cfg.device)
        self.num_actions = num_actions
        
        # Encoders: separate encoder for behavior policy μ to learn independently
        c, h, w = obs_shape
        self.encoder = CNNEncoder(obs_shape, embed_dim).to(self.device)      # For Q and π
        self.encoder_mu = CNNEncoder(obs_shape, embed_dim).to(self.device)   # For μ (BC)

        # Behavior policy μ (learns from data) and target policy π (optimized)
        self.mu = DiscretePolicy(embed_dim, num_actions, hidden).to(self.device)
        self.actor = DiscretePolicy(embed_dim, num_actions, hidden).to(self.device)

        # Critic with target network
        self.q = DiscreteQ(embed_dim, num_actions, hidden).to(self.device)
        self.q_tgt = DiscreteQ(embed_dim, num_actions, hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        # Optimizers
        # μ and its encoder train together via behavioral cloning
        self.opt_mu = torch.optim.Adam(
            list(self.mu.parameters()) + list(self.encoder_mu.parameters()), 
            lr=self.cfg.lr_critic
        )
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr_critic)
        self.opt_enc = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.lr_critic)

        # Hyperparameters
        self.lambda_kl = lambda_kl
        self.entropy_coef = entropy_coef
        self.bcq_threshold = bcq_threshold
        
        self.to(self.device)

    def update(self, batch):
        """
        Update agent from a batch of offline data
        
        4-stage update:
        1. Update behavior policy μ via behavioral cloning
        2. Update Q via TD learning
        3. Update target policy π with BRAC + BCQ constraints
        4. Update target network
        
        Args:
            batch: (S, A, R, S2, D) tuple of tensors
                
        Returns:
            Dict of training metrics
        """
        S, A, R, S2, D = [x.to(self.device) for x in batch]
        A = A.long().squeeze(-1)  # Shape: (B,) - squeeze in case it's (B, 1)
        D = D.float()

        # Encode states
        z = self.encoder(S)
        z2 = self.encoder(S2).detach()

        # ----- 1) Train behavior policy μ by behavioral cloning -----
        # μ learns to predict dataset actions via cross-entropy
        # Uses its own encoder so it can learn stable representations
        # Train multiple times to ensure μ keeps up with Q-function changes
        num_bc_updates = 3
        for _ in range(num_bc_updates):
            z_mu = self.encoder_mu(S)
            mu_pi, mu_log, mu_logits = self.mu(z_mu)
            mu_ce = -gather_qa(mu_log, A).mean()  # Negative log-likelihood

            self.opt_mu.zero_grad(set_to_none=True)
            mu_ce.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.mu.parameters()) + list(self.encoder_mu.parameters()), 
                self.cfg.grad_norm_clip
            )
            self.opt_mu.step()
        
        # Compute μ accuracy AFTER the updates
        with torch.no_grad():
            z_mu_new = self.encoder_mu(S)
            mu_pi_new, _, mu_logits_new = self.mu(z_mu_new)
            mu_greedy = mu_logits_new.argmax(-1)
            mu_acc = (mu_greedy == A).float().mean().item()

        # ----- 2) Critic update with TD target -----
        # TD target: y = r + γ * E_{a'~π}[Q_tgt(s',a')]
        with torch.no_grad():
            pi2, log_pi2, logits2 = self.actor(z2)
            
            # BCQ filtering: mask out unlikely actions
            if self.bcq_threshold is not None:
                z2_mu = self.encoder_mu(S2)  # Use μ's encoder for behavior policy
                mu2_prob, _, _ = self.mu(z2_mu)
                logits2 = mask_logits_with_mu(logits2, mu2_prob, self.bcq_threshold)
                log_pi2 = F.log_softmax(logits2, dim=-1)
                pi2 = log_pi2.exp()
            
            q2 = self.q_tgt(z2)
            v2 = (pi2 * q2).sum(-1)  # Expected value under π
            
            # Optional: add entropy bonus to target
            if self.entropy_coef > 0:
                ent2 = -(pi2 * log_pi2).sum(-1)
                v2 = v2 + self.entropy_coef * ent2
            
            y = R + (1.0 - D) * self.cfg.gamma * v2
            y = clamp_like(y, self.cfg.clamp_value_min, self.cfg.clamp_value_max)

        q = self.q(z)
        qa = gather_qa(q, A)
        td_loss = F.smooth_l1_loss(qa, y)
        
        # Q-value differentiation for discrete actions (standard adaptation)
        # Without this, Q-values collapse to same value for all actions
        # This is a soft CQL-like term: push up Q(data), push down Q(other)
        logsumexp_q = torch.logsumexp(q, dim=-1)
        q_reg = 0.5 * (logsumexp_q - qa).mean()
        
        critic_loss = td_loss + q_reg

        self.opt_q.zero_grad(set_to_none=True)
        self.opt_enc.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q.parameters()) + list(self.encoder.parameters()),
            self.cfg.grad_norm_clip
        )
        self.opt_q.step()
        self.opt_enc.step()

        # ----- 3) Actor update with BRAC + BCQ constraints -----
        # Objective: max E[Q(s,a)] - λ_KL * KL(π||μ) + λ_H * H(π)
        # Re-encode to avoid double backward
        z_actor = self.encoder(S)
        pi, log_pi, logits = self.actor(z_actor)
        
        # Get behavior policy from μ's encoder (detached - no gradient to μ)
        with torch.no_grad():
            z_mu_detached = self.encoder_mu(S)
            mu_prob, mu_log, _ = self.mu(z_mu_detached)
        
        # BCQ filtering for policy
        if self.bcq_threshold is not None:
            logits = mask_logits_with_mu(logits, mu_prob, self.bcq_threshold)
            log_pi = F.log_softmax(logits, dim=-1)
            pi = log_pi.exp()
        
        # Get Q-values for policy gradient
        q_for_pi = self.q(z_actor.detach()).detach()

        # Expected Q-value
        exp_q = (pi * q_for_pi).sum(-1)
        
        # BRAC: KL divergence penalty to stay close to behavior
        kl = kl_pi_mu(log_pi, mu_log)
        
        # Entropy bonus (optional)
        ent = -(pi * log_pi).sum(-1)

        # Combined objective
        actor_obj = exp_q - self.lambda_kl * kl + self.entropy_coef * ent
        actor_loss = -actor_obj.mean()

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_norm_clip)
        self.opt_actor.step()

        # ----- 4) Target network update -----
        polyak_update(self.q_tgt, self.q, self.cfg.tau)

        # ----- 5) Compute diagnostic metrics -----
        with torch.no_grad():
            # Mean Q across all actions
            mean_q = q.mean().item()
            # Q-value at dataset actions
            q_data = qa.mean().item()
            # Q-based action agreement (now meaningful with q_reg)
            greedy_a = q.argmax(-1)
            action_agree = (greedy_a == A).float().mean().item()

        return {
            "td_loss": float(td_loss.item()),
            "q_reg": float(q_reg.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mu_ce": float(mu_ce.item()),
            "kl": float(kl.mean().item()),
            "entropy": float(ent.mean().item()),
            "exp_q": float(exp_q.mean().item()),
            # Diagnostic metrics for logging
            "mean_q": mean_q,
            "q_data": q_data,
            "action_agree": action_agree,  # = μ accuracy for BRAC
            "mu_acc": mu_acc,  # Explicit μ accuracy (same as action_agree for BRAC)
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
        pi, log_pi, logits = self.actor(z)
        
        # Apply BCQ filtering if enabled (use μ's encoder)
        if self.bcq_threshold is not None:
            z_mu = self.encoder_mu(obs)
            mu_prob, _, _ = self.mu(z_mu)
            logits = mask_logits_with_mu(logits, mu_prob, self.bcq_threshold)
            log_pi = F.log_softmax(logits, dim=-1)
            pi = log_pi.exp()
        
        if eval_mode:
            # Greedy action
            action = pi.argmax(-1)
        else:
            # Sample from policy
            action = torch.multinomial(pi, num_samples=1).squeeze(-1)
        
        return action.cpu().numpy() if action.numel() > 1 else action.item()
