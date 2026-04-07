"""
Common utilities for offline discrete RL algorithms
Shared components across CQL, ERSAC, and other discrete methods
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlgoConfig:
    """Common configuration for offline RL algorithms"""
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Learning rates
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_alpha: float = 1e-4
    lr_value: float = 1e-4  # For IQL value network
    
    # RL hyperparameters
    gamma: float = 0.99
    tau: float = 0.005
    
    # Temperature
    init_alpha: float = 0.1
    target_entropy: Optional[float] = None  # If None, set to -num_actions
    
    # Training
    batch_size: int = 256
    grad_norm_clip: float = 10.0
    
    # Value clipping (for stability)
    clamp_value_min: float = -50.0
    clamp_value_max: float = 50.0


class DiscretePolicy(nn.Module):
    """Discrete policy network with softmax output"""
    
    def __init__(self, embed_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
        # Orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, embed_dim) encoded state
            
        Returns:
            pi: (B, A) action probabilities
            log_pi: (B, A) log probabilities
            logits: (B, A) raw logits (for BCQ masking)
        """
        logits = self.net(z)
        log_pi = F.log_softmax(logits, dim=-1)
        pi = log_pi.exp()
        return pi, log_pi, logits


class DiscreteQ(nn.Module):
    """Q-network for discrete actions"""
    
    def __init__(self, embed_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )
        # Orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim) encoded state
            
        Returns:
            q: (B, A) Q-values for all actions
        """
        return self.net(z)


class ValueNet(nn.Module):
    """Value network (scalar output for IQL)"""
    
    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, embed_dim) encoded state
            
        Returns:
            v: (B,) state values
        """
        return self.net(z).squeeze(-1)


def polyak_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """
    Polyak averaging for target network updates
    target = tau * source + (1 - tau) * target
    """
    with torch.no_grad():
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def clamp_like(tensor: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp tensor values while preserving gradient flow"""
    return torch.clamp(tensor, min=min_val, max=max_val)


def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """
    Asymmetric squared loss for expectile regression (used in IQL)
    
    L(u) = |τ - I(u < 0)| * u²
    
    Args:
        diff: (B,) or (B, A) differences (target - prediction)
        tau: expectile level in (0, 1). tau=0.5 is mean, tau>0.5 emphasizes upper tail
        
    Returns:
        loss: scalar loss value
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


class CNNEncoder(nn.Module):
    """CNN encoder for image observations (Atari-style)"""
    
    def __init__(self, obs_shape: Tuple[int, int, int], embed_dim: int = 512):
        super().__init__()
        c, h, w = obs_shape
        self.embed_dim = embed_dim  # Store for easy access
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).shape[1]
        
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh()
        )
        
        # Orthogonal initialization
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) image observations
            
        Returns:
            z: (B, embed_dim) encoded features
        """
        conv_out = self.conv(obs)
        return self.head(conv_out)
