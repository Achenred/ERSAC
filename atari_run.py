import minari

def ensure_datasets(dataset_list):
    for dataset_id in dataset_list:
        try:
            minari.load_dataset(dataset_id)
            print(f"Dataset '{dataset_id}' is already available locally.")
        except FileNotFoundError:
            print(f"Dataset '{dataset_id}' not found locally. Downloading...")
            minari.download_dataset(dataset_id)
            print(f"Dataset '{dataset_id}' downloaded.")

# Example usage:
datasets = [
    "atari/breakout/expert-v0",
    "atari/pong/expert-v0",
    # add more dataset IDs as needed
]
ensure_datasets(datasets)

# offline_sac_n_minari.py
# Implements Offline Discrete SAC-N for Atari pixels using a Minari dataset.
# Follows Eqs. (3)-(5) in "Epistemic Robust Offline Reinforcement Learning" (ICLR'26 under review).
# SAC-N here corresponds to ERSAC with a coordinate-wise box uncertainty set (Prop. 1).  [1]
#
# [1] Paper: "Epistemic Robust Offline Reinforcement Learning"
#     (see Sections 1.2, 2, Proposition 1, Algorithms A.5–A.6 for setup and training loops)

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

import minari  # pip install minari[all]  (you already have datasets locally)


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    dataset_id: str = "atari/breakout/expert-v0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Replay / stacking
    frame_size: Tuple[int, int] = (84, 84)  # H, W after resize
    frame_stack: int = 4

    # Model / training
    num_actions: int = 4  # will be inferred if not known; ALE/Breakout usually 4
    encoder_channels: Tuple[int, int, int] = (32, 64, 64)
    embed_dim: int = 512
    hidden_dim: int = 256

    # SAC-N
    ensemble_size: int = 5
    gamma: float = 0.99
    tau: float = 0.005            # Polyak for target
    init_alpha: float = 0.1       # entropy coefficient initial
    target_entropy: float = -1.0  # will set to -|A| by default in main

    # Optimization
    batch_size: int = 128
    lr_critic: float = 1e-4       # Lower LR for stability
    lr_actor: float = 1e-4        # Lower LR for stability
    lr_alpha: float = 1e-4        # Lower LR for stability
    total_steps: int = 200_000
    grad_norm_clip: float = 10.0
    
    # Update frequencies
    critic_update_freq: int = 1   # Update critic every N steps
    actor_update_freq: int = 2    # Update actor every N steps (slower = more stable)

    # Stability
    reward_clip: float = 1.0      # Clip rewards to [-reward_clip, reward_clip]
    normalize_rewards: bool = True # Normalize rewards by running stats
    use_huber_loss: bool = True   # Use Huber loss instead of MSE for critic (more robust)

    # Validation & Early Stopping
    use_validation: bool = True   # Split data into train/val for early stopping
    val_split: float = 0.1         # Fraction of data for validation
    eval_interval: int = 5000      # Evaluate on validation set every N steps
    patience: int = 3              # Stop if no improvement for N evaluations
    min_delta: float = 0.001       # Minimum improvement to count as better

    # Data loader - auto-detect optimal workers for the platform
    # Windows: conservative (4), Linux/Mac: more aggressive (8)
    num_workers: int = min(4, os.cpu_count() or 1) if os.name == 'nt' else min(8, os.cpu_count() or 1)
    pin_memory: bool = True
    persistent_workers: bool = True  # Keep workers alive between batches

    # Logging / eval
    log_interval: int = 1000
    eval_episodes: int = 10


# ----------------------------
# Minari -> Offline Replay builder (pixel + stacking)
# ----------------------------
class OfflineAtariDataset(Dataset):
    """
    Converts a Minari dataset of Atari episodes into (s, a, r, s', done) transitions
    with frame stacking (k=frame_stack). We build stacked states within each episode.
    Assumes observations are pixel arrays (H,W,3) or already grayscale; normalizes to [0,1].
    """

    def __init__(self, minari_dataset, frame_stack: int = 4, out_size=(84, 84), reward_clip: float = 1.0):
        super().__init__()
        self.ds = minari_dataset
        self.k = frame_stack
        self.H, self.W = out_size
        self.reward_clip = reward_clip
        self._build()

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale, resize to (H,W), and return float32 in [0,1].
        """
        # obs shape: (H0,W0,3) or (H0,W0)
        if obs.ndim == 3 and obs.shape[-1] == 3:
            # luminance
            obs = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
        obs = obs.astype(np.float32)

        # Resize using simple area interpolation (OpenCV not assumed). Use torch for resize once then back.
        # To avoid external deps, do a quick torch-based resize.
        ten = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ten = F.interpolate(ten, size=(self.H, self.W), mode="area")
        obs = ten.squeeze(0).squeeze(0).numpy()
        obs /= 255.0  # normalize
        return obs  # (H,W)

    def _stack(self, frames: List[np.ndarray]) -> np.ndarray:
        # frames: list of 2D (H,W), stack on channel => (k,H,W)
        return np.stack(frames, axis=0)

    def _build(self):
        S, A, R, S2, D = [], [], [], [], []
        for ep in self.ds.iterate_episodes():
            # ep.observations: list/array of observations; ep.actions; ep.rewards; dones implicit at end
            obs_list = ep.observations
            act_list = ep.actions
            rew_list = ep.rewards

            # preprocess all frames first
            proc = [self._preprocess_obs(o) for o in obs_list]  # length T+1 (if last obs is terminal)
            T = len(act_list)

            # build stacked frames within the episode
            frame_buffer = []
            for t in range(T + 1):
                frame_buffer.append(proc[t])
                if len(frame_buffer) > self.k:
                    frame_buffer.pop(0)
                # pad initial with first frame
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
                    # Clip rewards for stability
                    r = np.clip(r, -self.reward_clip, self.reward_clip)
                    s2 = s_stack
                    done = (t == T - 1)
                    S.append(s)
                    A.append(a)
                    R.append(r)
                    S2.append(s2)
                    D.append(done)

        self.S = np.asarray(S, dtype=np.float32)    # (N,k,H,W)
        self.A = np.asarray(A, dtype=np.int64)      # (N,)
        self.R = np.asarray(R, dtype=np.float32)    # (N,)
        self.S2 = np.asarray(S2, dtype=np.float32)  # (N,k,H,W)
        self.D = np.asarray(D, dtype=np.bool_)      # (N,)

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
# Networks: CNN encoder, Actor, Q-head
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
        # Lazy linear to infer conv shape at first forward
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(out_dim), nn.ReLU(inplace=True))
        self._init_conv_weights()
    
    def _init_conv_weights(self):
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B,k,H,W)
        x = self.conv(x)
        x = self.head(x)
        return x  # (B,out_dim)


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
        logits = self.net(embed)            # (B, A)
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
        return self.net(embed)  # (B, A)


# ----------------------------
# SAC-N Agent (discrete)
# ----------------------------
class SACNAgent:
    def __init__(self, cfg: Config, obs_shape: Tuple[int, int, int], num_actions: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_actions = num_actions

        c, h, w = obs_shape  # (k,H,W)

        # Shared encoder for actor and critics (common in vision RL)
        self.encoder = CNNEncoder(in_channels=c,
                                  ch=cfg.encoder_channels,
                                  out_dim=cfg.embed_dim).to(self.device)

        # Actor
        self.actor = PolicyNet(cfg.embed_dim, num_actions, hidden=cfg.hidden_dim).to(self.device)

        # Critic ensemble and targets
        self.q_nets = nn.ModuleList([QNet(cfg.embed_dim, num_actions, cfg.hidden_dim).to(self.device)
                                     for _ in range(cfg.ensemble_size)])
        self.q_targets = nn.ModuleList([QNet(cfg.embed_dim, num_actions, cfg.hidden_dim).to(self.device)
                                        for _ in range(cfg.ensemble_size)])
        # Init targets
        for q_tgt, q in zip(self.q_targets, self.q_nets):
            q_tgt.load_state_dict(q.state_dict())

        # Entropy temperature (alpha) learnable
        self.log_alpha = torch.tensor(math.log(cfg.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -float(num_actions) if cfg.target_entropy is None else cfg.target_entropy

        # Optims
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_encoder = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr_actor)  # share with actor update
        self.opt_q = torch.optim.Adam([p for q in self.q_nets for p in q.parameters()], lr=cfg.lr_critic)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        self.gamma = cfg.gamma
        self.tau = cfg.tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _min_q(self, q_list: List[torch.Tensor]) -> torch.Tensor:
        # q_list: list of (B,A) tensors
        q_stack = torch.stack(q_list, dim=0)   # (N,B,A)
        min_q, _ = torch.min(q_stack, dim=0)   # (B,A)
        return min_q

    def update(self, batch):
        S, A, R, S2, D = batch
        S = S.to(self.device)        # (B,k,H,W)
        A = A.to(self.device).long() # (B,)
        R = R.to(self.device)        # (B,)
        S2 = S2.to(self.device)      # (B,k,H,W)
        D = D.to(self.device).float() # (B,)

        # Encode
        z  = self.encoder(S)         # (B,E)
        z2 = self.encoder(S2).detach()

        # ------- Critic update (Eqs. (3)-(4)) -------
        with torch.no_grad():
            # Target policy on next state
            pi2, log_pi2, _ = self.actor(z2)
            # Target Q-values: min over ensemble (SAC-N conservative backup)
            q2_list = [q_tgt(z2) for q_tgt in self.q_targets]  # each (B,A)
            min_q2 = self._min_q(q2_list)                      # (B,A)
            # Soft value: E_{a'~pi}[ min_i Q_i(s',a') - alpha * log pi(a'|s') ]
            soft_v2 = (pi2 * (min_q2 - self.alpha * log_pi2)).sum(dim=-1)  # (B,)
            # Clip soft value for stability (since rewards are clipped to [-1, 1])
            soft_v2 = torch.clamp(soft_v2, -50.0, 50.0)
            y = R + (1.0 - D) * self.gamma * soft_v2               # (B,)
            # Clip target values for stability
            y = torch.clamp(y, -50.0, 50.0)

        # Current Q estimates
        q_list = [q(z) for q in self.q_nets]                  # each (B,A)
        # Gather on actions
        q_on_a = [qg.gather(1, A.unsqueeze(1)).squeeze(1) for qg in q_list]  # B each

        # Compute loss (Huber is more robust to outliers than MSE)
        critic_loss = 0.0
        if self.cfg.use_huber_loss:
            for q_a in q_on_a:
                critic_loss = critic_loss + F.smooth_l1_loss(q_a, y)  # Huber loss
        else:
            for q_a in q_on_a:
                critic_loss = critic_loss + F.mse_loss(q_a, y)
        critic_loss = critic_loss / len(q_on_a)

        self.opt_q.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for q in self.q_nets for p in q.parameters()],
                                       self.cfg.grad_norm_clip)
        self.opt_q.step()

        # ------- Actor + Encoder update (Eq. (5)) -------
        # Recompute z with encoder gradients enabled for joint actor+encoder update
        z_actor = self.encoder(S)               # forward pass for actor with encoder grads
        pi, log_pi, _ = self.actor(z_actor)
        q_for_pi = [q(z_actor) for q in self.q_nets]          # (B,A) each
        min_q = self._min_q(q_for_pi)                         # (B,A)

        # J_pi(ϕ) = E_s E_{a~pi}[ min_i Q_i(s,a) - alpha * log pi(a|s) ]
        actor_obj = (pi * (min_q - self.alpha * log_pi)).sum(dim=-1).mean()
        actor_loss = -actor_obj

        self.opt_actor.zero_grad(set_to_none=True)
        self.opt_encoder.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.encoder.parameters()),
                                       self.cfg.grad_norm_clip)
        self.opt_actor.step()
        self.opt_encoder.step()

        # ------- Temperature (alpha) update -------
        with torch.no_grad():
            # Recompute policy for alpha update (no gradients needed)
            z_for_alpha = self.encoder(S)
            pi_det, log_pi_det, _ = self.actor(z_for_alpha)
            # Discrete expectation E_{a~pi}[ - (log pi(a|s) + target_entropy) ]
            exp_term = (-(log_pi_det + self.target_entropy) * pi_det).sum(dim=-1)  # (B,)

        alpha_loss = (self.alpha * exp_term).mean()

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.opt_alpha.step()

        # ------- Target update -------
        with torch.no_grad():
            for q_tgt, q in zip(self.q_targets, self.q_nets):
                for p_t, p in zip(q_tgt.parameters(), q.parameters()):
                    p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return dict(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha=float(self.alpha.item()),
            alpha_loss=float(alpha_loss.item())
        )
    
    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        """Evaluate on validation set (no gradient updates)"""
        self.encoder.eval()
        self.actor.eval()
        [q.eval() for q in self.q_nets]
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            S, A, R, S2, D = batch
            S = S.to(self.device)
            A = A.to(self.device).long()
            R = R.to(self.device)
            S2 = S2.to(self.device)
            D = D.to(self.device).float()
            
            # Encode
            z = self.encoder(S)
            z2 = self.encoder(S2)
            
            # Target values
            pi2, log_pi2, _ = self.actor(z2)
            q2_list = [q_tgt(z2) for q_tgt in self.q_targets]
            min_q2 = self._min_q(q2_list)
            soft_v2 = (pi2 * (min_q2 - self.alpha * log_pi2)).sum(dim=-1)
            soft_v2 = torch.clamp(soft_v2, -50.0, 50.0)
            y = R + (1.0 - D) * self.gamma * soft_v2
            y = torch.clamp(y, -50.0, 50.0)
            
            # Current Q estimates
            q_list = [q(z) for q in self.q_nets]
            q_on_a = [qg.gather(1, A.unsqueeze(1)).squeeze(1) for qg in q_list]
            
            # Compute loss
            batch_loss = 0.0
            if self.cfg.use_huber_loss:
                for q_a in q_on_a:
                    batch_loss += F.smooth_l1_loss(q_a, y).item()
            else:
                for q_a in q_on_a:
                    batch_loss += F.mse_loss(q_a, y).item()
            batch_loss /= len(q_on_a)
            
            total_loss += batch_loss
            num_batches += 1
        
        # Restore training mode
        self.encoder.train()
        self.actor.train()
        [q.train() for q in self.q_nets]
        
        return total_loss / num_batches if num_batches > 0 else float('inf')


# ----------------------------
# Training entry
# ----------------------------
def train_offline_sac_n(cfg: Config):
    set_seed(cfg.seed)

    # Load dataset (no download if already local)
    ds = minari.load_dataset(cfg.dataset_id)  # will raise if missing; you already ensured availability
    
    # Infer action space
    action_space = ds.action_space
    if hasattr(action_space, "n"):
        num_actions = action_space.n
    else:
        # Fallback to config
        num_actions = cfg.num_actions

    print(f"Number of actions: {num_actions}")

    # Build replay dataset with stacking
    replay = OfflineAtariDataset(ds, frame_stack=cfg.frame_stack, out_size=cfg.frame_size, reward_clip=cfg.reward_clip)
    obs_shape = replay[0][0].shape  # (k,H,W)

    # Split into train/val if validation is enabled
    if cfg.use_validation:
        from torch.utils.data import random_split
        val_size = int(len(replay) * cfg.val_split)
        train_size = len(replay) - val_size
        train_dataset, val_dataset = random_split(replay, [train_size, val_size])
        
        use_persistent = cfg.num_workers > 0 and cfg.persistent_workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
        )
        print(f"DataLoader: {cfg.num_workers} workers, persistent={use_persistent}")
        loader = train_loader
        print(f"Dataset split - Train: {train_size}, Val: {val_size}")
    else:
        use_persistent = cfg.num_workers > 0 and cfg.persistent_workers
        loader = DataLoader(
            replay,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
        )
        val_loader = None
        print(f"Dataset size: {len(replay)} transitions (no validation split)")
        print(f"DataLoader: {cfg.num_workers} workers, persistent={use_persistent}")

    # Agent
    agent = SACNAgent(cfg, obs_shape=obs_shape, num_actions=num_actions)

    # Set target entropy default = -|A|
    agent.target_entropy = -float(num_actions)

    # Train
    agent.encoder.train(); agent.actor.train(); [q.train() for q in agent.q_nets]
    step = 0
    log_every = cfg.log_interval

    print(f"Starting training for {cfg.total_steps} steps...")
    print(f"Reward clipping: [-{cfg.reward_clip}, {cfg.reward_clip}]")
    print(f"Learning rates - Critic: {cfg.lr_critic}, Actor: {cfg.lr_actor}, Alpha: {cfg.lr_alpha}")
    print(f"Loss function: {'Huber (smooth_l1)' if cfg.use_huber_loss else 'MSE'}")
    
    # Early stopping tracking
    best_val_loss = float('inf')
    patience_counter = 0
    best_step = 0
    
    # Exponential moving average for smoother logging
    ema_alpha = 0.99
    ema_metrics = {}

    for epoch in range(10**9):  # until steps
        for batch in loader:
            metrics = agent.update(batch)
            step += 1
            
            # Update EMA metrics
            for k, v in metrics.items():
                if k not in ema_metrics:
                    ema_metrics[k] = v
                else:
                    ema_metrics[k] = ema_alpha * ema_metrics[k] + (1 - ema_alpha) * v

            # Validation & Early Stopping
            if cfg.use_validation and val_loader is not None and step % cfg.eval_interval == 0:
                val_loss = agent.evaluate(val_loader)
                print(f"[step {step:>6d}] Validation loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss - cfg.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_step = step
                    print(f"  ✓ New best validation loss!")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{cfg.patience})")
                    
                if patience_counter >= cfg.patience:
                    print(f"\nEarly stopping triggered! Best validation loss: {best_val_loss:.4f} at step {best_step}")
                    break

            if step % log_every == 0:
                print(f"[step {step:>6d}] "
                      f"critic_loss={ema_metrics['critic_loss']:>8.4f} (raw: {metrics['critic_loss']:>8.4f}) "
                      f"actor_loss={ema_metrics['actor_loss']:>8.4f} "
                      f"alpha={ema_metrics['alpha']:>6.4f}")

            if step >= cfg.total_steps:
                break
        
        # Check if early stopping was triggered
        if cfg.use_validation and patience_counter >= cfg.patience:
            break
            
        if step >= cfg.total_steps:
            break

    print("Training complete.")
    if cfg.use_validation:
        print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
    return agent


if __name__ == "__main__":
    cfg = Config(
        dataset_id="atari/breakout/expert-v0",
        total_steps=100_000,         # adjust as needed
        batch_size=128,
        ensemble_size=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_offline_sac_n(cfg)