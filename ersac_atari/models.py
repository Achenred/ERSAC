
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fanin_init(tensor, fanin=None):
    fanin = fanin or tensor.size(0)
    bound = 1. / math.sqrt(fanin)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class AtariEncoder(nn.Module):
    """
    DQN-style CNN encoder for Atari RGB observations in (B, C, H, W), pixels in [0,1].
    """
    def __init__(self, in_channels: int, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self._proj = None
        self.out_dim = out_dim

    def _build_proj(self, x: torch.Tensor):
        with torch.no_grad():
            h = self.conv(x)
            nflat = h.view(h.size(0), -1).size(1)
        self._proj = nn.Sequential(
            nn.Linear(nflat, self.out_dim),
            nn.ReLU(inplace=True),
        )
        for m in self._proj:
            if isinstance(m, nn.Linear):
                fanin_init(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._proj is None:
            dummy = torch.zeros_like(x[:1])
            self._build_proj(dummy)
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self._proj(h)
        return z


class DiscretePolicy(nn.Module):
    """SAC policy for discrete actions: outputs logits for softmax policy."""
    def __init__(self, feat_dim: int, n_actions: int):
        super().__init__()
        self.head = nn.Linear(feat_dim, n_actions)
        fanin_init(self.head.weight)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    @staticmethod
    def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        pi = torch.softmax(logits, dim=-1)
        logpi = torch.log_softmax(logits, dim=-1)
        return -(pi * logpi).sum(dim=-1)


class QHead(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int):
        super().__init__()
        self.head = nn.Linear(feat_dim, n_actions)
        fanin_init(self.head.weight)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class QEnsemble(nn.Module):
    """Ensemble of Q-heads (SAC-N) sharing a common encoder for efficiency."""
    def __init__(self, encoder: AtariEncoder, feat_dim: int, n_actions: int, n_members: int):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([QHead(feat_dim, n_actions) for _ in range(n_members)])
        self.n_members = n_members

    def forward_all(self, obs: torch.Tensor) -> List[torch.Tensor]:
        feat = self.encoder(obs)
        return [h(feat) for h in self.heads]

    def forward_min(self, obs: torch.Tensor) -> torch.Tensor:
        qs = self.forward_all(obs)  # list of (B, A)
        q_stack = torch.stack(qs, dim=0)  # (N, B, A)
        return q_stack.min(dim=0).values  # (B, A)


@dataclass
class EpinetOut:
    mu: torch.Tensor        # (B, A)
    Sigma: torch.Tensor     # (B, A, A)


class EpinetQ(nn.Module):
    """
    Epinet critic with linear stochastic head in z (Assumption 3):
      q(s,·,z) = mu(s,·) + <sigma_bar(s,·), z>,  z ~ N(0, I_{d_z})
    where sigma_bar(s,·) \in R^{A x d_z}. Then Cov[q] = Sigma = sigma_bar @ sigma_bar^T.
    """
    def __init__(self, encoder: AtariEncoder, feat_dim: int, n_actions: int, z_dim: int = 16,
                 prior_scale: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.mu_head = nn.Linear(feat_dim, n_actions)
        fanin_init(self.mu_head.weight)
        # learnable linear map to A*d_z, reshape to (A, d_z)
        self.sigma_L = nn.Linear(feat_dim, n_actions * z_dim)
        fanin_init(self.sigma_L.weight)
        # fixed prior map (frozen params)
        self.sigma_P = nn.Linear(feat_dim, n_actions * z_dim, bias=False)
        with torch.no_grad():
            self.sigma_P.weight.normal_(std=prior_scale)
        for p in self.sigma_P.parameters():
            p.requires_grad_(False)
        self.n_actions = n_actions
        self.z_dim = z_dim

    def forward_mu_sigma_bar(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(obs)
        mu = self.mu_head(feat)  # (B, A)
        sL = self.sigma_L(feat).view(feat.size(0), self.n_actions, self.z_dim)
        sP = self.sigma_P(feat).view(feat.size(0), self.n_actions, self.z_dim)
        sigma_bar = sL + sP  # (B, A, d_z)
        return mu, sigma_bar

    def forward(self, obs: torch.Tensor) -> EpinetOut:
        mu, sigma_bar = self.forward_mu_sigma_bar(obs)
        # Sigma = sigma_bar @ sigma_bar^T over z-dim
        Sigma = torch.einsum('bad,bcd->bac', sigma_bar, sigma_bar.transpose(1,2))  # (B, A, A)
        return EpinetOut(mu=mu, Sigma=Sigma)

    def sample_q(self, obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        mu, sigma_bar = self.forward_mu_sigma_bar(obs)
        B, A, dz = sigma_bar.shape
        z = torch.randn(n_samples, B, dz, device=obs.device)
        # (n, B, A, dz) x (n, B, dz) -> (n, B, A)
        qb = torch.einsum('bad,nbd->nba', sigma_bar, z)
        return mu.unsqueeze(0) + qb
