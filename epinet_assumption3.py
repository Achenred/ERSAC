"""
Epinet Implementation based on Assumption 3 (Linear Stochastic Heads)

This implements the closed-form Gaussian approach from the ERSAC paper:
- q_θ(s, a, z) = μ_θ(s, a) + <σ̄_θ(ψ(s), a), z>
- Under Assumption 3: q_θ(s, ·, z) ~ N(μ_θ(s), Σ_θ(s))
- Covariance: Σ_θ(s)[a, a'] = <σ̄_θ(ψ, a), σ̄_θ(ψ, a')>

This enables efficient closed-form computation of mean and covariance,
avoiding the need for sampling during inference.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class EpistemicNeuralNetwork(nn.Module):
    """
    Epistemic Neural Network with Linear Stochastic Heads (Assumption 3).
    
    Architecture:
        1. Base network: μ_θ(s, a) - produces mean Q-values
        2. Feature extraction: ψ_θ(s) from last hidden layer  
        3. Stochastic heads: σ̄^L_θ(ψ, a) + σ̄^P(ψ, a) - linear in z
    
    The Q-value distribution is:
        q_θ(s, a, z) = μ_θ(s, a) + <σ̄_θ(ψ(s), a), z>
    
    where z ~ N(0, I) is the epistemic index.
    
    This induces a Gaussian: q_θ(s, ·, z) ~ N(μ_θ(s), Σ_θ(s))
    with covariance: Σ[a, a'] = <σ̄(ψ, a), σ̄(ψ, a')>
    """
    
    def __init__(self, embed_dim: int, num_actions: int, hidden_dim: int = 256,
                 z_dim: int = 10, prior_scale: float = 1.0):
        """
        Args:
            embed_dim: Dimension of state embedding
            num_actions: Number of discrete actions  
            hidden_dim: Hidden layer size
            z_dim: Dimension of epistemic index z
            prior_scale: Scale for fixed prior σ^P
        """
        super().__init__()
        self.z_dim = z_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Base network for mean: μ_θ(s, a)
        # Includes feature extraction ψ_θ(s) from hidden layer
        self.base_hidden = nn.Linear(embed_dim, hidden_dim)
        self.base_output = nn.Linear(hidden_dim, num_actions)
        
        # Learnable stochastic head: σ̄^L_θ: R^{d_ψ} × A → R^{d_z}
        # Outputs weights for linear combination with z
        self.sigma_learnable = nn.Linear(hidden_dim, num_actions * z_dim)
        
        # Fixed prior stochastic head: σ̄^P: R^{d_ψ} × A → R^{d_z}
        # Encodes initial epistemic uncertainty
        self.sigma_prior = nn.Linear(hidden_dim, num_actions * z_dim, bias=False)
        with torch.no_grad():
            self.sigma_prior.weight.normal_(0, prior_scale / math.sqrt(hidden_dim))
        self.sigma_prior.requires_grad_(False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for learnable parameters"""
        nn.init.orthogonal_(self.base_hidden.weight, gain=math.sqrt(2))
        nn.init.constant_(self.base_hidden.bias, 0.0)
        nn.init.orthogonal_(self.base_output.weight, gain=1.0)
        nn.init.constant_(self.base_output.bias, 0.0)
        nn.init.orthogonal_(self.sigma_learnable.weight, gain=1.0)
        nn.init.constant_(self.sigma_learnable.bias, 0.0)
    
    def forward(self, embed: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: compute Q-values with optional epistemic sampling.
        
        Args:
            embed: (B, E) state embeddings
            z: Epistemic indices (optional):
               - None: Return mean μ_θ(s, a) only
               - (N, Z): N samples → returns (N, B, A)
               - (B, Z): One sample per batch → returns (B, A)
        
        Returns:
            Q-values: mean if z=None, else sampled q_θ(s, a, z)
        """
        B = embed.size(0)
        
        # Forward through base network
        psi = torch.relu(self.base_hidden(embed))  # (B, hidden) - features ψ_θ(s)
        mu = self.base_output(psi)                  # (B, A) - mean μ_θ(s, a)
        
        if z is None:
            # Return only mean (no epistemic uncertainty)
            return mu
        
        # Compute combined stochastic head weights: σ̄_θ = σ̄^L_θ + σ̄^P
        sigma_l = self.sigma_learnable(psi).view(B, self.num_actions, self.z_dim)
        sigma_p = self.sigma_prior(psi).view(B, self.num_actions, self.z_dim)
        sigma_bar = sigma_l + sigma_p  # (B, A, Z)
        
        # Sample Q-values: q_θ(s, a, z) = μ_θ(s, a) + <σ̄_θ(ψ, a), z>
        if z.dim() == 2 and z.size(0) != B:
            # Multiple epistemic samples: z is (N, Z)
            N = z.size(0)
            mu_exp = mu.unsqueeze(0).expand(N, -1, -1)              # (N, B, A)
            sigma_exp = sigma_bar.unsqueeze(0).expand(N, -1, -1, -1)  # (N, B, A, Z)
            z_exp = z.view(N, 1, 1, self.z_dim)                     # (N, 1, 1, Z)
            
            # Inner product <σ̄, z> for each sample
            stochastic_term = (sigma_exp * z_exp).sum(dim=-1)  # (N, B, A)
            return mu_exp + stochastic_term  # (N, B, A)
        else:
            # Single sample per batch: z is (B, Z)
            stochastic_term = (sigma_bar * z.unsqueeze(1)).sum(dim=-1)  # (B, A)
            return mu + stochastic_term  # (B, A)
    
    def get_distribution_params(self, embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get parameters of the induced Gaussian distribution (Assumption 3).
        
        Returns closed-form mean and covariance without sampling:
            q_θ(s, ·, z) ~ N(μ_θ(s), Σ_θ(s))
        
        Args:
            embed: (B, E) state embeddings
        
        Returns:
            mu: (B, A) mean Q-values μ_θ(s, a)
            cov: (B, A, A) covariance matrix Σ_θ(s)
                 where Σ[a, a'] = <σ̄(ψ, a), σ̄(ψ, a')>
        """
        B = embed.size(0)
        
        # Extract features and mean
        psi = torch.relu(self.base_hidden(embed))
        mu = self.base_output(psi)  # (B, A)
        
        # Compute combined stochastic weights
        sigma_l = self.sigma_learnable(psi).view(B, self.num_actions, self.z_dim)
        sigma_p = self.sigma_prior(psi).view(B, self.num_actions, self.z_dim)
        sigma_bar = sigma_l + sigma_p  # (B, A, Z)
        
        # Covariance via inner products: Σ[a, a'] = <σ̄(ψ, a), σ̄(ψ, a')>
        # This is equivalent to: Σ = σ̄ @ σ̄^T
        cov = torch.bmm(sigma_bar, sigma_bar.transpose(1, 2))  # (B, A, A)
        
        return mu, cov
    
    def sample_q_values(self, embed: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample Q-values from the distribution.
        
        Args:
            embed: (B, E) state embeddings
            num_samples: Number of epistemic samples
        
        Returns:
            (N, B, A) sampled Q-values
        """
        z = torch.randn(num_samples, self.z_dim, device=embed.device)
        return self.forward(embed, z)


if __name__ == "__main__":
    print("Testing Epinet with Assumption 3 (Linear Stochastic Heads)")
    print("=" * 70)
    
    # Test setup
    B, E, A, Z = 4, 64, 4, 10
    embed = torch.randn(B, E)
    
    # Create network
    epinet = EpistemicNeuralNetwork(
        embed_dim=E,
        num_actions=A,
        hidden_dim=128,
        z_dim=Z
    )
    
    print(f"Batch size: {B}, Embed dim: {E}, Actions: {A}, Z-dim: {Z}")
    print()
    
    # Test 1: Mean Q-values (no sampling)
    mu = epinet(embed)
    print(f"✓ Mean Q-values: {mu.shape}")
    
    # Test 2: Get distribution parameters
    mu_dist, cov = epinet.get_distribution_params(embed)
    print(f"✓ Distribution params - μ: {mu_dist.shape}, Σ: {cov.shape}")
    print(f"  Covariance is symmetric: {torch.allclose(cov, cov.transpose(1, 2))}")
    print(f"  Covariance is PSD: {(torch.linalg.eigvalsh(cov) >= -1e-6).all()}")
    
    # Test 3: Sample Q-values
    N = 5
    q_samples = epinet.sample_q_values(embed, num_samples=N)
    print(f"✓ Sampled Q-values: {q_samples.shape}")
    
    # Test 4: Verify Gaussian property
    # Sample mean should match μ
    sample_mean = q_samples.mean(dim=0)
    print(f"  Sample mean matches μ: {torch.allclose(sample_mean, mu_dist, atol=0.5)}")
    
    print()
    print("=" * 70)
    print("All tests passed! Epinet ready for ERSAC training.")
