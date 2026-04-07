"""
ERSAC Training Script with Multiple Uncertainty Set Options
Based on "Epistemic Robust Offline Reinforcement Learning" (ICLR'26 under review)

Supports:
- Box uncertainty set (standard SAC-N)
- Convex Hull uncertainty set
- Ellipsoidal uncertainty set (sample-based)
- Epinet with ellipsoidal uncertainty set
"""

import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import minari

# Import ERSAC extended components
from ersac_agent_extended import ERSACAgentExtended, ERSACConfig


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_datasets(dataset_list):
    for dataset_id in dataset_list:
        try:
            minari.load_dataset(dataset_id)
            print(f"Dataset '{dataset_id}' is already available locally.")
        except FileNotFoundError:
            print(f"Dataset '{dataset_id}' not found locally. Downloading...")
            minari.download_dataset(dataset_id)
            print(f"Dataset '{dataset_id}' downloaded.")


# ----------------------------
# Dataset (from atari_run.py)
# ----------------------------
class OfflineAtariDataset(Dataset):
    """
    Converts a Minari dataset of Atari episodes into (s, a, r, s', done) transitions
    with frame stacking (k=frame_stack).
    """

    def __init__(self, minari_dataset, frame_stack: int = 4, out_size=(84, 84), reward_clip: float = 1.0):
        super().__init__()
        self.ds = minari_dataset
        self.k = frame_stack
        self.H, self.W = out_size
        self.reward_clip = reward_clip
        self._build()

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert to grayscale, resize to (H,W), and return float32 in [0,1]."""
        if obs.ndim == 3 and obs.shape[-1] == 3:
            obs = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
        obs = obs.astype(np.float32)

        ten = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ten = F.interpolate(ten, size=(self.H, self.W), mode="area")
        obs = ten.squeeze(0).squeeze(0).numpy()
        obs /= 255.0
        return obs

    def _stack(self, frames: List[np.ndarray]) -> np.ndarray:
        return np.stack(frames, axis=0)

    def _build(self):
        S, A, R, S2, D = [], [], [], [], []
        for ep in self.ds.iterate_episodes():
            obs_list = ep.observations
            act_list = ep.actions
            rew_list = ep.rewards

            proc = [self._preprocess_obs(o) for o in obs_list]
            T = len(act_list)

            frame_buffer = []
            for t in range(T + 1):
                frame_buffer.append(proc[t])
                if len(frame_buffer) > self.k:
                    frame_buffer.pop(0)
                while len(frame_buffer) < self.k:
                    frame_buffer.insert(0, frame_buffer[0])

                if t == 0:
                    s_stack = self._stack(frame_buffer)
                    prev_stack = s_stack
                else:
                    prev_stack = s_stack
                    s_stack = self._stack(frame_buffer)

                if t < T:
                    s = prev_stack
                    a = act_list[t]
                    r = float(rew_list[t])
                    r = np.clip(r, -self.reward_clip, self.reward_clip)
                    s2 = s_stack
                    done = (t == T - 1)
                    S.append(s)
                    A.append(a)
                    R.append(r)
                    S2.append(s2)
                    D.append(done)

        self.S = np.asarray(S, dtype=np.float32)
        self.A = np.asarray(A, dtype=np.int64)
        self.R = np.asarray(R, dtype=np.float32)
        self.S2 = np.asarray(S2, dtype=np.float32)
        self.D = np.asarray(D, dtype=np.bool_)

    def __len__(self):
        return self.S.shape[0]

    def __getitem__(self, idx):
        return (
            self.S[idx],    # (k,H,W)
            self.A[idx],    # ()
            self.R[idx],    # ()
            self.S2[idx],   # (k,H,W)
            self.D[idx],    # ()
        )


# ----------------------------
# Networks (from atari_run.py)
# ----------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, ch=(32, 64, 64), out_dim=512):
        super().__init__()
        c1, c2, c3 = ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(out_dim), nn.ReLU(inplace=True))
        self._init_conv_weights()
    
    def _init_conv_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, embed_dim, num_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, embed):
        return self.net(embed)


# ----------------------------
# Training Function
# ----------------------------
def train_ersac(uncertainty_set: str = "box", use_epinet: bool = False):
    """
    Train ERSAC with specified uncertainty set method.
    
    Args:
        uncertainty_set: "box", "convex_hull", "ellipsoidal", or "epinet"
        use_epinet: If True and uncertainty_set=="epinet", use Epinet architecture
    """
    
    # Configuration
    dataset_id = "atari/breakout/expert-v0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    set_seed(seed)
    
    # Ensure dataset is available
    ensure_datasets([dataset_id])
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_id}")
    ds = minari.load_dataset(dataset_id)
    
    # Infer action space
    action_space = ds.action_space
    if hasattr(action_space, "n"):
        num_actions = action_space.n
    else:
        num_actions = 4
    
    print(f"Number of actions: {num_actions}")
    
    # Create ERSAC configuration
    ersac_cfg = ERSACConfig(
        uncertainty_set=uncertainty_set,
        use_epinet=use_epinet,
        num_actions=num_actions,
        ensemble_size=5,
        gamma=0.99,
        tau=0.005,
        init_alpha=0.1,
        lr_critic=1e-4,
        lr_actor=1e-4,
        lr_alpha=1e-4,
        grad_norm_clip=10.0,
        # Ellipsoidal parameters
        ellipsoid_coverage=0.9,
        min_variance=1e-6,
        # Epinet parameters
        epinet_z_dim=10,
        epinet_prior_scale=1.0,
        bootstrap_noise_scale=0.1,
        epinet_l2_mu=0.0,
        epinet_l2_sigma=0.0,
    )
    
    # Build replay dataset
    print(f"\nBuilding offline dataset...")
    replay = OfflineAtariDataset(
        ds, 
        frame_stack=4, 
        out_size=(84, 84), 
        reward_clip=1.0
    )
    obs_shape = replay[0][0].shape  # (k,H,W)
    print(f"Dataset size: {len(replay)} transitions")
    print(f"Observation shape: {obs_shape}")
    
    # Create dataloader
    loader = DataLoader(
        replay,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    # Create encoder
    print(f"\nCreating encoder and agent...")
    encoder = CNNEncoder(
        in_channels=obs_shape[0],
        ch=(32, 64, 64),
        out_dim=512
    ).to(device)
    
    # Create ERSAC agent
    agent = ERSACAgentExtended(
        cfg=ersac_cfg,
        obs_shape=obs_shape,
        num_actions=num_actions,
        encoder=encoder,
        device=device
    )
    
    # Set target entropy
    agent.target_entropy = -float(num_actions)
    
    # Training
    print(f"\n{'='*60}")
    print(f"Starting Training with {uncertainty_set.upper()} uncertainty set")
    if use_epinet:
        print(f"Using Epinet architecture")
    print(f"{'='*60}")
    print(f"Reward clipping: [-1.0, 1.0]")
    print(f"Learning rates - Critic: {ersac_cfg.lr_critic}, Actor: {ersac_cfg.lr_actor}")
    print(f"Uncertainty set: {uncertainty_set}")
    
    # Set networks to training mode
    agent.encoder.train()
    agent.actor.train()
    if use_epinet:
        agent.q_net.train()
    else:
        [q.train() for q in agent.q_nets]
    
    step = 0
    total_steps = 100_000
    log_every = 1000
    
    # Exponential moving average for smoother logging
    ema_alpha = 0.99
    ema_metrics = {}
    
    for epoch in range(10**9):
        for batch in loader:
            metrics = agent.update(batch)
            step += 1
            
            # Update EMA metrics
            for k, v in metrics.items():
                if k not in ema_metrics:
                    ema_metrics[k] = v
                else:
                    ema_metrics[k] = ema_alpha * ema_metrics[k] + (1 - ema_alpha) * v
            
            if step % log_every == 0:
                print(f"[step {step:>6d}] "
                      f"critic_loss={ema_metrics['critic_loss']:>8.4f} (raw: {metrics['critic_loss']:>8.4f}) "
                      f"actor_loss={ema_metrics['actor_loss']:>8.4f} "
                      f"alpha={ema_metrics['alpha']:>6.4f}")
            
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    return agent


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import sys
    
    print("ERSAC Training Script")
    print("="*60)
    print("Available uncertainty sets:")
    print("  1. box          - Standard SAC-N (coordinate-wise min)")
    print("  2. convex_hull  - Worst consistent Q-network")
    print("  3. ellipsoidal  - Ellipsoidal uncertainty (sample-based)")
    print("  4. epinet       - Epinet with ellipsoidal (most efficient)")
    print("="*60)
    
    # Choose method (you can change this or make it a command-line argument)
    if len(sys.argv) > 1:
        uncertainty_set = sys.argv[1]
    else:
        # Default to box for comparison with original
        uncertainty_set = "box"
    
    use_epinet = (uncertainty_set == "epinet")
    
    print(f"\nTraining with: {uncertainty_set.upper()}")
    print()
    
    try:
        agent = train_ersac(uncertainty_set=uncertainty_set, use_epinet=use_epinet)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()