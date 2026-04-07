"""
Extended ERSAC Agent with multiple uncertainty set options.
Integrates with the existing atari_run.py architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass

try:
    from ersac_variants import (
        BoxUncertaintySet,
        ConvexHullUncertaintySet,
        EllipsoidalUncertaintySet,
        EpinetEllipsoidalUncertaintySet,
        EpistemicNeuralNetwork
    )
except ImportError as e:
    print(f"Error importing ersac_variants: {e}")
    print("Make sure ersac_variants.py is in the same directory and all dependencies are installed.")
    print("You may need to run: pip install scipy")
    raise


@dataclass
class ERSACConfig:
    """Extended configuration for ERSAC variants"""
    # Uncertainty set type
    uncertainty_set: str = "box"  # Options: "box", "convex_hull", "ellipsoidal", "epinet"
    
    # Ellipsoidal set parameters
    ellipsoid_coverage: float = 0.9  # Coverage proportion (υ in paper)
    min_variance: float = 1e-6  # Minimum variance for stability
    
    # Epinet parameters
    use_epinet: bool = False  # Use Epinet instead of ensemble
    epinet_z_dim: int = 10  # Dimension of epistemic index
    epinet_prior_scale: float = 1.0
    bootstrap_noise_scale: float = 0.1  # For ensemble diversity (Gaussian bootstrap)
    
    # Behavioral cloning for discrete actions
    bc_weight: float = 1.0  # Weight for BC loss in actor update (critical for discrete offline RL)
    
    # Entropy parameters for discrete actions
    target_entropy_ratio: float = 0.5  # Target entropy = ratio * log(num_actions)
    learn_alpha: bool = True  # If False, alpha stays fixed at init_alpha
    
    # Base SAC parameters (from original Config)
    num_actions: int = 4
    ensemble_size: int = 5
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.1
    
    # Optimization
    lr_critic: float = 1e-4
    lr_actor: float = 1e-4
    lr_alpha: float = 1e-4
    grad_norm_clip: float = 10.0
    
    # Regularization for Epinet
    epinet_l2_mu: float = 0.0
    epinet_l2_sigma: float = 0.0
    
    # EDAC-style diversity regularization
    # Encourages ensemble disagreement to maintain exploration signal
    edac_eta: float = 0.0  # EDAC diversity coefficient (0 = disabled, try 1.0-10.0)


class ERSACAgentExtended:
    """
    Extended ERSAC agent supporting multiple uncertainty set constructions.
    Compatible with the architecture from atari_run.py.
    """
    
    def __init__(self, cfg: ERSACConfig, obs_shape, num_actions: int, 
                 encoder: nn.Module, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_actions = num_actions
        
        # Shared encoder (passed from outside)
        self.encoder = encoder
        
        # Get embed_dim from encoder
        embed_dim = getattr(encoder, 'embed_dim', 512)
        
        # Import network classes from main_run (they'll be there)
        try:
            from main_run import PolicyNet, QNet
        except ImportError:
            # Fallback: define minimal versions
            class PolicyNet(nn.Module):
                def __init__(self, embed_dim, num_actions, hidden=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
                        nn.Linear(hidden, num_actions)
                    )
                def forward(self, embed):
                    logits = self.net(embed)
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    return probs, log_probs, logits
            
            class QNet(nn.Module):
                def __init__(self, embed_dim, num_actions, hidden=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
                        nn.Linear(hidden, num_actions)
                    )
                def forward(self, embed):
                    return self.net(embed)
        
        # Actor
        self.actor = PolicyNet(embed_dim, num_actions, hidden=256).to(self.device)
        
        # Initialize Q-networks based on configuration
        if cfg.use_epinet:
            # Use single Epinet instead of ensemble
            self.q_net = EpistemicNeuralNetwork(
                embed_dim=embed_dim,
                num_actions=num_actions,
                hidden_dim=256,
                z_dim=cfg.epinet_z_dim,
                prior_scale=cfg.epinet_prior_scale
            ).to(self.device)
            
            self.q_target = EpistemicNeuralNetwork(
                embed_dim=embed_dim,
                num_actions=num_actions,
                hidden_dim=256,
                z_dim=cfg.epinet_z_dim,
                prior_scale=cfg.epinet_prior_scale
            ).to(self.device)
            self.q_target.load_state_dict(self.q_net.state_dict())
            
            self.opt_q = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr_critic)
        else:
            # Use ensemble of Q-networks (original approach)
            self.q_nets = nn.ModuleList([
                QNet(embed_dim, num_actions, 256).to(self.device)
                for _ in range(cfg.ensemble_size)
            ])
            self.q_targets = nn.ModuleList([
                QNet(embed_dim, num_actions, 256).to(self.device)
                for _ in range(cfg.ensemble_size)
            ])
            for q_tgt, q in zip(self.q_targets, self.q_nets):
                q_tgt.load_state_dict(q.state_dict())
            
            self.opt_q = torch.optim.Adam(
                [p for q in self.q_nets for p in q.parameters()],
                lr=cfg.lr_critic
            )
        
        # Initialize uncertainty set method
        self.uncertainty_set = self._create_uncertainty_set(cfg)
        
        # Optimizers
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr_actor)
        
        # Entropy temperature (handle init_alpha=0 case)
        init_alpha_safe = max(cfg.init_alpha, 1e-8)  # Avoid log(0)
        self.log_alpha = torch.tensor(
            math.log(init_alpha_safe), device=self.device, requires_grad=True
        )
        # For discrete actions, target entropy = ratio * log(num_actions)
        # Max entropy for uniform over n actions is log(n)
        # Using ratio < 1 allows some specialization while maintaining exploration
        max_entropy = math.log(num_actions)
        self.target_entropy = cfg.target_entropy_ratio * max_entropy
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        
        self.gamma = cfg.gamma
        self.tau = cfg.tau
    
    def _create_uncertainty_set(self, cfg):
        """Create appropriate uncertainty set based on config"""
        if cfg.uncertainty_set == "box":
            return BoxUncertaintySet()
        elif cfg.uncertainty_set == "convex_hull":
            return ConvexHullUncertaintySet(alpha=cfg.init_alpha)
        elif cfg.uncertainty_set == "ellipsoidal":
            return EllipsoidalUncertaintySet(
                coverage=cfg.ellipsoid_coverage,
                min_variance=cfg.min_variance
            )
        elif cfg.uncertainty_set == "epinet" and cfg.use_epinet:
            return EpinetEllipsoidalUncertaintySet(
                coverage=cfg.ellipsoid_coverage,
                num_actions=cfg.num_actions
            )
        else:
            raise ValueError(f"Unknown uncertainty set: {cfg.uncertainty_set}")
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def _get_q_values(self, z, is_target=False):
        """Get Q-values from appropriate network(s)"""
        if self.cfg.use_epinet:
            # Sample from Epinet
            N_samples = self.cfg.ensemble_size
            z_samples = torch.randn(N_samples, self.cfg.epinet_z_dim, device=self.device)
            
            if is_target:
                q_samples = self.q_target(z, z_samples)  # (N, B, A)
            else:
                q_samples = self.q_net(z, z_samples)  # (N, B, A)
            
            # Convert to list for compatibility
            return [q_samples[i] for i in range(N_samples)]
        else:
            # Use ensemble
            if is_target:
                return [q_tgt(z) for q_tgt in self.q_targets]
            else:
                return [q(z) for q in self.q_nets]
    
    def update(self, batch):
        """
        Update agent - extended version supporting multiple uncertainty sets.
        """
        S, A, R, S2, D = batch
        S = S.to(self.device)
        A = A.to(self.device).long()
        R = R.to(self.device)
        S2 = S2.to(self.device)
        D = D.to(self.device).float()
        
        # Encode
        z = self.encoder(S)
        z2 = self.encoder(S2).detach()
        
        # ------- Critic update -------
        with torch.no_grad():
            # Target policy on next state
            pi2, log_pi2, _ = self.actor(z2)
            
            # Update alpha in convex hull uncertainty set (for proper worst-case selection)
            if isinstance(self.uncertainty_set, ConvexHullUncertaintySet):
                self.uncertainty_set.set_alpha(self.alpha.item())
            
            # Get target Q-values (ensemble or Epinet)
            q2_list = self._get_q_values(z2, is_target=True)
            
            # Compute worst-case Q using selected uncertainty set
            if isinstance(self.uncertainty_set, EpinetEllipsoidalUncertaintySet) and self.cfg.use_epinet:
                # Use closed-form Epinet computation
                mu2, cov2 = self.q_target.get_distribution_params(z2)
                min_q2 = self.uncertainty_set.compute_worst_case_q_from_params(mu2, cov2, pi2)
            elif isinstance(self.uncertainty_set, ConvexHullUncertaintySet):
                # Pass log_pi for proper soft Q selection
                min_q2 = self.uncertainty_set.compute_worst_case_q(q2_list, pi2, log_pi2)
            else:
                # Box and Ellipsoidal don't need log_pi
                min_q2 = self.uncertainty_set.compute_worst_case_q(q2_list, pi2)
            
            # Soft value
            soft_v2 = (pi2 * (min_q2 - self.alpha * log_pi2)).sum(dim=-1)
            soft_v2 = torch.clamp(soft_v2, -50.0, 50.0)
            y = R + (1.0 - D) * self.gamma * soft_v2
            y = torch.clamp(y, -50.0, 50.0)
        
        # Current Q estimates
        if self.cfg.use_epinet:
            # Epinet training with bootstrap
            N_samples = self.cfg.ensemble_size
            z_samples = torch.randn(N_samples, self.cfg.epinet_z_dim, device=self.device)
            
            # Perturbed targets for Epinet (Gaussian bootstrap)
            c = torch.randn(S.size(0), self.cfg.epinet_z_dim, device=self.device)
            c = c / c.norm(dim=-1, keepdim=True)  # Normalize to unit sphere
            
            # Sample Q-values
            q_samples = self.q_net(z, z_samples)  # (N, B, A)
            
            # Gather on actions and compute loss with bootstrap
            critic_loss = 0.0
            for i in range(N_samples):
                q_a = q_samples[i].gather(1, A.unsqueeze(1)).squeeze(1)
                bootstrap_noise = self.cfg.bootstrap_noise_scale * (c * z_samples[i].unsqueeze(0)).sum(dim=-1)
                target = y + bootstrap_noise
                critic_loss += F.smooth_l1_loss(q_a, target)
            critic_loss = critic_loss / N_samples
            
            # Add L2 regularization
            if self.cfg.epinet_l2_mu > 0 or self.cfg.epinet_l2_sigma > 0:
                l2_loss = 0.0
                for name, param in self.q_net.named_parameters():
                    if 'base_net' in name or 'feature_net' in name:
                        l2_loss += self.cfg.epinet_l2_mu * (param ** 2).sum()
                    elif 'stochastic_head' in name:
                        l2_loss += self.cfg.epinet_l2_sigma * (param ** 2).sum()
                critic_loss += l2_loss
        else:
            # Standard ensemble training with Gaussian bootstrap (Equation 10 in paper)
            # Each network gets independent target: y_i = y + η_i where η_i ~ N(0, σ²)
            q_list = self._get_q_values(z, is_target=False)
            q_on_a = [qg.gather(1, A.unsqueeze(1)).squeeze(1) for qg in q_list]
            
            critic_loss = 0.0
            for i, q_a in enumerate(q_on_a):
                # Add independent noise for ensemble diversity (Gaussian bootstrap)
                bootstrap_noise = self.cfg.bootstrap_noise_scale * torch.randn_like(y)
                y_i = y + bootstrap_noise
                critic_loss += F.smooth_l1_loss(q_a, y_i)
            # DON'T divide by N - each network should get full gradient
            # (Averaging makes gradients vanish for large ensembles)
            
            # EDAC-style diversity regularization
            # Encourages ensemble members to disagree, maintaining uncertainty signal
            # Loss = -η * std(Q_i(s,a)) across ensemble
            if self.cfg.edac_eta > 0:
                q_stack = torch.stack(q_on_a, dim=0)  # (N, B)
                q_std = q_stack.std(dim=0)  # (B,) - std across ensemble
                # Maximize std by minimizing -std (gradient ascent on diversity)
                diversity_loss = -self.cfg.edac_eta * q_std.mean()
                critic_loss = critic_loss + diversity_loss
        
        self.opt_q.zero_grad(set_to_none=True)
        self.opt_encoder.zero_grad(set_to_none=True)  # Train encoder with critic too
        critic_loss.backward()
        
        # Compute gradient norms BEFORE clipping for diagnostics
        if self.cfg.use_epinet:
            q_params = list(self.q_net.parameters())
        else:
            q_params = [p for q in self.q_nets for p in q.parameters()]
        
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(q_params, float('inf')).item()
        encoder_grad_norm_critic = torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()), float('inf')
        ).item()
        
        # Now apply actual clipping
        if self.cfg.use_epinet:
            torch.nn.utils.clip_grad_norm_(
                list(self.q_net.parameters()) + list(self.encoder.parameters()),
                self.cfg.grad_norm_clip
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for q in self.q_nets for p in q.parameters()] + list(self.encoder.parameters()),
                self.cfg.grad_norm_clip
            )
        self.opt_q.step()
        self.opt_encoder.step()  # Update encoder with critic
        
        # ------- Actor update (encoder DETACHED to prevent overfitting) -------
        # Detach encoder - only train actor MLP, not encoder (prevents memorization)
        z_actor = self.encoder(S).detach()  # Detach to prevent overfitting
        pi, log_pi, logits = self.actor(z_actor)
        
        # Get current Q-values
        q_for_pi_list = self._get_q_values(z_actor, is_target=False)
        q_for_pi = [q.detach() for q in q_for_pi_list]
        
        # IMPORTANT: Based on tabular ERSAC implementation (VaRFramework/src/algorithms.py),
        # the ACTOR always uses BOX (element-wise min) regardless of uncertainty set.
        # The uncertainty set (ellipsoidal, convex hull) is used for the CRITIC TARGET,
        # but the actor uses min_i Q_i(s,a) for each action independently.
        # 
        # This makes sense because:
        # 1. Actor wants to maximize Q - we want pessimistic Q per action
        # 2. Ellipsoidal worst-case is for expected value under policy (used in target)
        # 3. Box gives clear action ranking (min across ensemble for each action)
        if self.cfg.use_epinet:
            # For epinet, get the mean Q directly
            mu_actor, _ = self.q_net.get_distribution_params(z_actor)
            min_q = mu_actor  # Use mean Q for actor (pessimism is in the target)
        else:
            # Standard ensemble: use element-wise min (box) for actor
            q_stack = torch.stack(q_for_pi, dim=0)  # (N, B, A)
            min_q = q_stack.min(dim=0)[0]  # (B, A) - box/min across ensemble
        
        # Store min_q statistics for diagnostics
        min_q_mean = min_q.mean().item()
        min_q_std_across_actions = min_q.std(dim=-1).mean().item()  # How much Q varies across actions
        
        # ERSAC actor objective (Algorithm 3 in paper)
        # Paper: ∑_a q*(s,a) · ∇π(a|s) - α·∇E[log π]
        # 
        # Standard form: maximize E_π[Q - α·log π]
        actor_obj = (pi * (min_q.detach() - self.alpha * log_pi)).sum(dim=-1).mean()
        
        # Behavioral cloning loss: directly imitate expert actions
        bc_loss = F.nll_loss(log_pi, A)  # Negative log-likelihood of expert actions
        
        # Combined loss: ERSAC objective + BC regularization
        actor_loss = -actor_obj + self.cfg.bc_weight * bc_loss
        
        self.opt_actor.zero_grad(set_to_none=True)
        # DON'T update encoder with actor/BC - causes overfitting!
        actor_loss.backward()
        
        # Compute gradient norms BEFORE clipping for diagnostics
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()), float('inf')
        ).item()
        # Encoder grad from actor is computed but NOT applied
        encoder_grad_norm_actor = 0.0  # Not updating encoder with actor anymore
        
        # Now apply actual clipping - ONLY actor
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()),
            self.cfg.grad_norm_clip
        )
        self.opt_actor.step()
        # NO encoder step here - only critic updates encoder
        
        # ------- Alpha update (only if learn_alpha=True) -------
        if self.cfg.learn_alpha:
            with torch.no_grad():
                z_for_alpha = self.encoder(S)
                pi_det, log_pi_det, _ = self.actor(z_for_alpha)
                exp_term = (-(log_pi_det + self.target_entropy) * pi_det).sum(dim=-1)
            
            alpha_loss = (self.alpha * exp_term).mean()
            
            self.opt_alpha.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.opt_alpha.step()
        else:
            alpha_loss = torch.tensor(0.0)  # No update
        
        # ------- Target update -------
        with torch.no_grad():
            if self.cfg.use_epinet:
                for p_t, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            else:
                for q_tgt, q in zip(self.q_targets, self.q_nets):
                    for p_t, p in zip(q_tgt.parameters(), q.parameters()):
                        p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        
        # Store BC loss value before diagnostics
        bc_loss_val = bc_loss.item()
        
        # ------- Compute diagnostic metrics -------
        with torch.no_grad():
            # Get Q-values for diagnostics
            z_diag = self.encoder(S)
            q_diag_list = self._get_q_values(z_diag, is_target=False)
            q_mean = torch.stack(q_diag_list, dim=0).mean(dim=0)  # (B, A)
            
            # Mean Q across all actions
            mean_q = q_mean.mean().item()
            
            # Q-value at data actions
            q_data = q_mean.gather(1, A.unsqueeze(1)).squeeze(1).mean().item()
            
            # Action agreement: does actor's greedy policy match expert data?
            pi_diag, log_pi_diag, _ = self.actor(z_diag)
            actor_greedy_actions = pi_diag.argmax(dim=-1)
            action_agree = (actor_greedy_actions == A).float().mean().item()
            
            # Entropy of current policy
            entropy = -(pi_diag * log_pi_diag).sum(dim=-1).mean().item()
            
            # Q-value spread across ensemble (uncertainty measure)
            q_stack = torch.stack(q_diag_list, dim=0)  # (N, B, A)
            q_std = q_stack.std(dim=0).mean().item()
        
        return dict(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            bc_loss=bc_loss_val,
            alpha=float(self.alpha.item()),
            alpha_loss=float(alpha_loss.item()),
            mean_q=mean_q,
            q_data=q_data,
            action_agree=action_agree,
            entropy=entropy,
            q_std=q_std,
            min_q_std=min_q_std_across_actions,  # How much worst-case Q varies across actions
            # Gradient norms for debugging
            critic_grad=critic_grad_norm,
            encoder_grad_c=encoder_grad_norm_critic,
            actor_grad=actor_grad_norm,
            encoder_grad_a=encoder_grad_norm_actor,
        )


def print_usage_example():
    """Print example usage"""
    example_code = """
# Example Usage:

from ersac_agent_extended import ERSACAgentExtended, ERSACConfig

# 1. Box-based (standard SAC-N)
config_box = ERSACConfig(
    uncertainty_set="box",
    ensemble_size=5,
    num_actions=4
)

# 2. Convex Hull
config_ch = ERSACConfig(
    uncertainty_set="convex_hull",
    ensemble_size=5,
    num_actions=4
)

# 3. Ellipsoidal (sample-based)
config_ell = ERSACConfig(
    uncertainty_set="ellipsoidal",
    ellipsoid_coverage=0.9,  # 90% coverage
    ensemble_size=5,
    num_actions=4
)

# 4. Epinet with Ellipsoidal
config_epinet = ERSACConfig(
    uncertainty_set="epinet",
    use_epinet=True,
    epinet_z_dim=10,
    ellipsoid_coverage=0.9,
    bootstrap_noise_scale=0.1,
    num_actions=4
)

# Create agent (after creating encoder)
agent = ERSACAgentExtended(
    cfg=config,
    obs_shape=(4, 84, 84),
    num_actions=4,
    encoder=encoder,  # Your CNN encoder
    device="cuda"
)

# Training loop (same as before)
for batch in loader:
    metrics = agent.update(batch)
    """
    print(example_code)


if __name__ == "__main__":
    print("ERSAC Extended Agent Module")
    print("="*50)
    print("Supports:")
    print("  1. Box uncertainty set (standard SAC-N)")
    print("  2. Convex Hull uncertainty set")
    print("  3. Ellipsoidal uncertainty set (sample-based)")
    print("  4. Epinet with ellipsoidal uncertainty set")
    print("\n")
    print_usage_example()
