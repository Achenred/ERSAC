"""
ERSAC Variants: Convex Hull, Ellipsoidal, and Epinet-based methods
Extends the box-based SAC-N with additional uncertainty set constructions
from "Epistemic Robust Offline Reinforcement Learning" (ICLR'26 under review)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from scipy.stats import chi2


# ----------------------------
# Uncertainty Set Methods
# ----------------------------

class UncertaintySetMethod:
    """Base class for uncertainty set construction methods"""
    
    def compute_worst_case_q(self, q_list: List[torch.Tensor], pi: torch.Tensor) -> torch.Tensor:
        """
        Compute worst-case Q-values given ensemble predictions and policy.
        
        Args:
            q_list: List of (B, A) tensors - Q-values from ensemble
            pi: (B, A) tensor - policy probabilities
            
        Returns:
            (B, A) tensor - worst-case Q-values
        """
        raise NotImplementedError


class BoxUncertaintySet(UncertaintySetMethod):
    """
    Box (coordinate-wise min) uncertainty set - this is standard SAC-N.
    U_box(s) = ×_a [min_i Q_i(s,a), max_i Q_i(s,a)]
    """
    
    def compute_worst_case_q(self, q_list: List[torch.Tensor], pi: torch.Tensor) -> torch.Tensor:
        """Element-wise minimum across ensemble (policy-independent)"""
        q_stack = torch.stack(q_list, dim=0)  # (N, B, A)
        min_q, _ = torch.min(q_stack, dim=0)  # (B, A)
        return min_q


class ConvexHullUncertaintySet(UncertaintySetMethod):
    """
    Convex hull uncertainty set - finds worst consistent Q-function.
    U_hull(s) = {Σ_i λ_i Q_i(s,·) : λ ≥ 0, Σ λ_i = 1}
    
    For discrete actions, the worst-case is the Q-network with lowest
    expected value under the current policy.
    """
    
    def __init__(self, alpha: float = 0.0):
        """
        Args:
            alpha: Entropy coefficient (needed for proper worst-case selection)
        """
        super().__init__()
        self.alpha = alpha
    
    def set_alpha(self, alpha: float):
        """Update alpha (called from agent when alpha changes)"""
        self.alpha = alpha
    
    def compute_worst_case_q(self, q_list: List[torch.Tensor], pi: torch.Tensor, 
                             log_pi: torch.Tensor = None) -> torch.Tensor:
        """
        Find the Q-network with worst expected value under policy.
        
        This is the closed-form solution for discrete actions:
        q*(s,·) = Q_{i*}(s,·) where i* = argmin_i E_a~π[Q_i(s,a) - α log π(a|s)]
        
        Args:
            q_list: List of Q-values from ensemble networks
            pi: Policy probabilities (B, A)
            log_pi: Log policy probabilities (B, A) - needed for proper selection
        """
        q_stack = torch.stack(q_list, dim=0)  # (N, B, A)
        
        # Expected soft Q-value under policy for each network: (N, B)
        # Paper Eq: argmin_i E_π[Q_i - α log π]
        if log_pi is not None and self.alpha > 0:
            soft_q = q_stack - self.alpha * log_pi.unsqueeze(0)  # (N, B, A)
            expected_soft_q = (soft_q * pi.unsqueeze(0)).sum(dim=-1)  # (N, B)
        else:
            # Without entropy, just expected Q
            expected_soft_q = (q_stack * pi.unsqueeze(0)).sum(dim=-1)  # (N, B)
        
        # Find worst network per batch element: (B,)
        worst_idx = torch.argmin(expected_soft_q, dim=0)
        
        # Gather Q-values from worst network for each batch element
        batch_size = q_stack.size(1)
        batch_indices = torch.arange(batch_size, device=q_stack.device)
        q_worst = q_stack[worst_idx, batch_indices, :]  # (B, A)
        
        return q_worst


class EllipsoidalUncertaintySet(UncertaintySetMethod):
    """
    Ellipsoidal uncertainty set based on empirical mean and covariance.
    U_ell(s) = {q : (q - μ)^T Σ^{-1} (q - μ) ≤ Υ²}
    
    The worst-case Q-vector has closed form:
    q*(s,·) = μ(s) - Υ(s) · Σ(s)π(·|s) / ||Σ^{1/2}(s)π(·|s)||
    """
    
    def __init__(self, coverage: float = 0.9, min_variance: float = 1e-6):
        """
        Args:
            coverage: Proportion of empirical mass to cover (υ in paper)
            min_variance: Minimum variance for numerical stability
        """
        super().__init__()
        self.coverage = coverage
        self.min_variance = min_variance
    
    def compute_worst_case_q(self, q_list: List[torch.Tensor], pi: torch.Tensor) -> torch.Tensor:
        """
        Compute worst-case Q using ellipsoidal uncertainty set.
        """
        q_stack = torch.stack(q_list, dim=0)  # (N, B, A)
        N, B, A = q_stack.shape
        
        # Compute empirical mean: (B, A)
        mu = q_stack.mean(dim=0)
        
        # Compute empirical covariance: (B, A, A)
        centered = q_stack - mu.unsqueeze(0)  # (N, B, A)
        # Reshape for batch matrix multiplication
        centered_t = centered.transpose(0, 1)  # (B, N, A)
        cov = torch.bmm(centered_t.transpose(1, 2), centered_t) / N  # (B, A, A)
        
        # Add small regularization for numerical stability
        cov = cov + self.min_variance * torch.eye(A, device=cov.device).unsqueeze(0)
        
        # Compute radius Υ based on coverage
        # For each batch element, find radius that covers coverage proportion
        Upsilon_sq = self._compute_radius(q_stack, mu, cov, self.coverage)  # (B,)
        
        # Compute worst-case Q: μ - Υ · Σπ / ||Σ^{1/2}π||
        # Solve: Σπ = cov @ pi
        Sigma_pi = torch.bmm(cov, pi.unsqueeze(-1)).squeeze(-1)  # (B, A)
        
        # Compute ||Σ^{1/2}π||²  = π^T Σ π
        pi_Sigma_pi = torch.bmm(pi.unsqueeze(1), Sigma_pi.unsqueeze(-1)).squeeze()  # (B,)
        norm_Sigma_half_pi = torch.sqrt(torch.clamp(pi_Sigma_pi, min=1e-8))  # (B,)
        
        # Compute worst-case: μ - Υ · Σπ / ||Σ^{1/2}π||
        Upsilon = torch.sqrt(Upsilon_sq)  # (B,)
        q_worst = mu - (Upsilon / norm_Sigma_half_pi).unsqueeze(-1) * Sigma_pi  # (B, A)
        
        return q_worst
    
    def _compute_radius(self, q_stack: torch.Tensor, mu: torch.Tensor, 
                       cov: torch.Tensor, coverage: float) -> torch.Tensor:
        """
        Compute radius Υ² such that P((q-μ)^T Σ^{-1} (q-μ) ≤ Υ²) ≥ coverage
        
        Args:
            q_stack: (N, B, A)
            mu: (B, A)
            cov: (B, A, A)
            coverage: target coverage proportion
            
        Returns:
            (B,) tensor of radius² values
        """
        N, B, A = q_stack.shape
        
        # Compute Mahalanobis distance for each sample
        centered = q_stack - mu.unsqueeze(0)  # (N, B, A)
        centered_t = centered.transpose(0, 1)  # (B, N, A)
        
        # Solve cov @ x = centered for each sample
        # distances[b, i] = (q_i - μ)^T Σ^{-1} (q_i - μ)
        try:
            cov_inv = torch.linalg.inv(cov)  # (B, A, A)
            # For each batch and sample: centered[b, i]^T @ cov_inv[b] @ centered[b, i]
            temp = torch.bmm(centered_t, cov_inv)  # (B, N, A)
            distances = (temp * centered_t).sum(dim=-1)  # (B, N)
        except:
            # Fallback if inversion fails
            distances = (centered_t ** 2).sum(dim=-1)  # (B, N)
        
        # For each batch element, find the coverage-quantile
        Upsilon_sq = torch.quantile(distances, coverage, dim=1)  # (B,)
        
        return Upsilon_sq


class EpinetEllipsoidalUncertaintySet(UncertaintySetMethod):
    """
    Ellipsoidal uncertainty set using Epinet's structured representation.
    Instead of sampling, uses closed-form Gaussian distribution.
    
    Assumes Q-distribution is: q ~ N(μ_θ(s), Σ_θ(s))
    where Σ is computed from the Epinet stochastic heads.
    """
    
    def __init__(self, coverage: float = 0.9, num_actions: int = 4):
        """
        Args:
            coverage: Confidence level (υ in paper)
            num_actions: Number of actions
        """
        super().__init__()
        self.coverage = coverage
        self.num_actions = num_actions
        # Compute radius from chi-squared distribution
        self.Upsilon_sq = chi2.ppf(coverage, df=num_actions)
    
    def compute_worst_case_q_from_params(self, mu: torch.Tensor, cov: torch.Tensor, 
                                         pi: torch.Tensor) -> torch.Tensor:
        """
        Compute worst-case Q directly from mean and covariance.
        
        Args:
            mu: (B, A) - mean Q-values
            cov: (B, A, A) - covariance matrix
            pi: (B, A) - policy probabilities
            
        Returns:
            (B, A) - worst-case Q-values
        """
        # Compute Σπ
        Sigma_pi = torch.bmm(cov, pi.unsqueeze(-1)).squeeze(-1)  # (B, A)
        
        # Compute ||Σ^{1/2}π||
        pi_Sigma_pi = torch.bmm(pi.unsqueeze(1), Sigma_pi.unsqueeze(-1)).squeeze()  # (B,)
        norm_Sigma_half_pi = torch.sqrt(torch.clamp(pi_Sigma_pi, min=1e-8))  # (B,)
        
        # Worst-case Q
        Upsilon = math.sqrt(self.Upsilon_sq)
        q_worst = mu - (Upsilon / norm_Sigma_half_pi).unsqueeze(-1) * Sigma_pi  # (B, A)
        
        return q_worst


# ----------------------------
# Epinet Architecture
# ----------------------------

class EpistemicNeuralNetwork(nn.Module):
    """
    Epistemic Neural Network (Epinet) with Linear Stochastic Heads (Assumption 3).
    
    Following the paper, this implements:
        q_θ(s, a, z) = μ_θ(s, a) + <σ̄_θ(ψ(s), a), z>
    
    where:
        - μ_θ(s, a): Base network (mean Q-values)
        - ψ(s): Feature representation from last hidden layer
        - σ̄_θ(ψ, a) = σ̄^L_θ(ψ, a) + σ̄^P(ψ, a): Combined stochastic weights
        - z ~ N(0, I): Epistemic index
    
    Under Assumption 3 (linear in z), this induces a Gaussian distribution:
        q_θ(s, ·, z) ~ N(μ_θ(s), Σ_θ(s))
    with covariance: Σ[a, a'] = <σ̄(ψ, a), σ̄(ψ, a')>
    """
    
    def __init__(self, embed_dim: int, num_actions: int, hidden_dim: int = 256,
                 z_dim: int = 10, prior_scale: float = 1.0):
        """
        Args:
            embed_dim: Dimension of state embedding
            num_actions: Number of discrete actions
            hidden_dim: Hidden layer size (for ψ features)
            z_dim: Dimension of epistemic index z
            prior_scale: Scale for fixed prior σ^P
        """
        super().__init__()
        self.z_dim = z_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Shared hidden layer for features ψ_θ(s)
        self.feature_layer = nn.Linear(embed_dim, hidden_dim)
        
        # Base network head: μ_θ(s, a) = f(ψ_θ(s))
        self.mu_head = nn.Linear(hidden_dim, num_actions)
        
        # Learnable stochastic head: σ̄^L_θ(ψ, a) ∈ R^{d_z}
        # Outputs weights for linear combination with z
        self.sigma_learnable = nn.Linear(hidden_dim, num_actions * z_dim)
        
        # Fixed prior stochastic head: σ̄^P(ψ, a) ∈ R^{d_z}
        self.sigma_prior = nn.Linear(hidden_dim, num_actions * z_dim, bias=False)
        with torch.no_grad():
            self.sigma_prior.weight.normal_(0, prior_scale / math.sqrt(hidden_dim))
        self.sigma_prior.requires_grad_(False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for learnable parameters"""
        nn.init.orthogonal_(self.feature_layer.weight, gain=math.sqrt(2))
        nn.init.constant_(self.feature_layer.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=1.0)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.orthogonal_(self.sigma_learnable.weight, gain=1.0)
        nn.init.constant_(self.sigma_learnable.bias, 0.0)
    
    def forward(self, embed: torch.Tensor, z: Optional[torch.Tensor] = None):
        """
        Forward pass: compute Q-values with optional epistemic sampling.
        
        Args:
            embed: (B, E) state embedding
            z: Epistemic indices (optional):
               - None: Return mean μ_θ(s, a) only
               - (N, Z): N samples → returns (N, B, A)
               - (B, Z): One sample per batch → returns (B, A)
            
        Returns:
            Q-values: mean if z=None, else sampled q_θ(s, a, z)
        """
        B = embed.size(0)
        
        # Extract features ψ_θ(s) and compute mean μ_θ(s, a)
        psi = torch.relu(self.feature_layer(embed))  # (B, hidden)
        mu = self.mu_head(psi)                        # (B, A)
        
        if z is None:
            return mu
        
        # Compute combined stochastic weights: σ̄_θ = σ̄^L_θ + σ̄^P
        sigma_l = self.sigma_learnable(psi).view(B, self.num_actions, self.z_dim)
        sigma_p = self.sigma_prior(psi).view(B, self.num_actions, self.z_dim)
        sigma_bar = sigma_l + sigma_p  # (B, A, Z)
        
        # Sample Q-values: q_θ(s, a, z) = μ_θ(s, a) + <σ̄_θ(ψ, a), z>
        if z.dim() == 2 and z.size(0) != B:
            # Multiple epistemic samples: z is (N, Z)
            N = z.size(0)
            mu_exp = mu.unsqueeze(0).expand(N, -1, -1)                # (N, B, A)
            sigma_exp = sigma_bar.unsqueeze(0).expand(N, -1, -1, -1)  # (N, B, A, Z)
            z_exp = z.view(N, 1, 1, self.z_dim)                       # (N, 1, 1, Z)
            
            stochastic_term = (sigma_exp * z_exp).sum(dim=-1)  # (N, B, A)
            return mu_exp + stochastic_term
        else:
            # Single sample per batch: z is (B, Z)
            stochastic_term = (sigma_bar * z.unsqueeze(1)).sum(dim=-1)  # (B, A)
            return mu + stochastic_term
    
    def get_distribution_params(self, embed: torch.Tensor):
        """
        Get parameters of the induced Gaussian distribution (Assumption 3).
        
        Returns closed-form mean and covariance:
            q_θ(s, ·, z) ~ N(μ_θ(s), Σ_θ(s))
        where: Σ[a, a'] = <σ̄(ψ, a), σ̄(ψ, a')>
        
        Args:
            embed: (B, E) state embedding
            
        Returns:
            mu: (B, A) mean Q-values
            cov: (B, A, A) covariance matrix
        """
        B = embed.size(0)
        
        # Extract features and mean
        psi = torch.relu(self.feature_layer(embed))
        mu = self.mu_head(psi)  # (B, A)
        
        # Compute combined stochastic weights
        sigma_l = self.sigma_learnable(psi).view(B, self.num_actions, self.z_dim)
        sigma_p = self.sigma_prior(psi).view(B, self.num_actions, self.z_dim)
        sigma_bar = sigma_l + sigma_p  # (B, A, Z)
        
        # Covariance: Σ = σ̄ @ σ̄^T
        cov = torch.bmm(sigma_bar, sigma_bar.transpose(1, 2))  # (B, A, A)
        
        return mu, cov


print("ERSAC Variants module loaded successfully!")
print("Available uncertainty sets:")
print("  - BoxUncertaintySet (standard SAC-N)")
print("  - ConvexHullUncertaintySet")
print("  - EllipsoidalUncertaintySet")
print("  - EpinetEllipsoidalUncertaintySet")
print("  - EpistemicNeuralNetwork (Epinet architecture)")
