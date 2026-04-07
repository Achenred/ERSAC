# benchmark_comparison.py
"""
Benchmark comparison: CQL, IQL, BRAC-BCQ, and ERSAC variants
Compare different offline RL methods on Atari environments
"""

import os
import sys
import argparse
import torch
import minari
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_optimal_num_workers(device: str) -> int:
    """Get optimal number of DataLoader workers based on system."""
    if device != "cuda":
        return 0  # CPU-only: no benefit from workers
    
    # On Windows, multiprocessing can be problematic; use conservative value
    if os.name == 'nt':  # Windows
        return min(4, os.cpu_count() or 1)
    else:  # Linux/Mac
        return min(8, os.cpu_count() or 1)

from common_offline_discrete import AlgoConfig, CNNEncoder
from cql_discrete import CQLAgent
from iql_discrete import IQLAgent
from brac_bcq_discrete import BRACAgentDiscrete
from ersac_agent_extended import ERSACAgentExtended, ERSACConfig


class OfflineAtariDataset(Dataset):
    """Dataset wrapper for Minari Atari datasets"""
    
    def __init__(self, dataset_name: str, max_episodes: int = None):
        self.dataset = minari.load_dataset(dataset_name)
        self.episodes = list(self.dataset.iterate_episodes(max_episodes))
        self._build()
    
    def _preprocess_obs(self, obs):
        """Convert observation to (C, H, W) float tensor"""
        if isinstance(obs, dict) and 'pixels' in obs:
            obs = obs['pixels']
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[-1] in [1, 3]:
            obs = obs.transpose(2, 0, 1)
        return obs / 255.0
    
    def _stack(self, frames):
        """Stack k frames along channel dimension"""
        k = 4
        if len(frames) < k:
            frames = [frames[0]] * (k - len(frames)) + frames
        else:
            frames = frames[-k:]
        return np.concatenate(frames, axis=0)
    
    def _build(self):
        """Build dataset from episodes"""
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones = [], []
        
        for ep in self.episodes:
            obs_list = [self._preprocess_obs(o) for o in ep.observations]
            
            for t in range(len(ep.actions)):
                if t == 0:
                    s = self._stack([obs_list[0]])
                else:
                    s = self._stack(obs_list[max(0, t-3):t+1])
                
                if t+1 < len(obs_list):
                    s2 = self._stack(obs_list[max(0, t-2):t+2])
                else:
                    s2 = s
                
                # Clip rewards to [-1, 1]
                r = np.clip(ep.rewards[t], -1.0, 1.0)
                
                self.states.append(s)
                self.actions.append(ep.actions[t])
                self.rewards.append(r)
                self.next_states.append(s2)
                self.dones.append(ep.terminations[t] or ep.truncations[t])
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]),
            torch.tensor(self.actions[idx], dtype=torch.long),
            torch.tensor(self.rewards[idx], dtype=torch.float32),
            torch.from_numpy(self.next_states[idx]),
            torch.tensor(self.dones[idx], dtype=torch.float32)
        )


def train_agent(agent, dataloader, num_steps: int, method_name: str):
    """
    Train an agent for a specified number of steps
    
    Args:
        agent: CQLAgent or ERSACAgentExtended
        dataloader: DataLoader for offline data
        num_steps: Number of training steps
        method_name: Name of the method for logging
    
    Returns:
        agent: Trained agent
    """
    print(f"\n{'='*60}")
    print(f"Training {method_name}")
    print(f"{'='*60}")
    
    agent_iter = iter(dataloader)
    ema_critic = None
    ema_actor = None
    
    for step in range(1, num_steps + 1):
        # Get batch
        try:
            batch = next(agent_iter)
        except StopIteration:
            agent_iter = iter(dataloader)
            batch = next(agent_iter)
        
        # Update agent
        metrics = agent.update(batch)
        
        # EMA smoothing for logging
        if ema_critic is None:
            ema_critic = metrics['critic_loss']
            ema_actor = metrics['actor_loss']
        else:
            ema_critic = 0.99 * ema_critic + 0.01 * metrics['critic_loss']
            ema_actor = 0.99 * ema_actor + 0.01 * metrics['actor_loss']
        
        # Logging
        if step % 1000 == 0:
            log_str = f"[{method_name}] Step {step:6d} | "
            log_str += f"Critic: {ema_critic:8.2f} | "
            log_str += f"Actor: {ema_actor:8.2f}"
            
            if 'alpha' in metrics:
                log_str += f" | Alpha: {metrics['alpha']:.3f}"
            
            if 'cql_pen' in metrics:
                log_str += f" | CQL: {metrics['cql_pen']:8.2f}"
            
            if 'value_loss' in metrics:
                log_str += f" | Value: {metrics['value_loss']:8.2f}"
            
            if 'mean_advantage' in metrics:
                log_str += f" | Adv: {metrics['mean_advantage']:6.2f}"
            
            if 'kl' in metrics:
                log_str += f" | KL: {metrics['kl']:6.3f}"
            
            if 'mu_ce' in metrics:
                log_str += f" | BC: {metrics['mu_ce']:6.3f}"
            
            print(log_str)
    
    print(f"\n{method_name} training complete!")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Benchmark CQL, IQL, BRAC-BCQ, and ERSAC variants")
    parser.add_argument("method", type=str, 
                       choices=["cql", "iql", "brac", "box", "convex_hull", "ellipsoidal", "epinet"],
                       help="Method to train")
    parser.add_argument("--cql_alpha", type=float, default=1.0,
                       help="CQL penalty weight (only for CQL)")
    parser.add_argument("--expectile_tau", type=float, default=0.7,
                       help="Expectile level for IQL (only for IQL, 0.7 = 70th percentile)")
    parser.add_argument("--beta_adv", type=float, default=3.0,
                       help="Advantage temperature for IQL (only for IQL)")
    parser.add_argument("--lambda_kl", type=float, default=1.0,
                       help="KL penalty weight for BRAC (only for BRAC)")
    parser.add_argument("--bcq_threshold", type=float, default=None,
                       help="BCQ filtering threshold (only for BRAC, None = no filtering)")
    parser.add_argument("--dataset", type=str, default="atari/breakout/expert-v0",
                       help="Minari dataset name")
    parser.add_argument("--steps", type=int, default=100000,
                       help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = OfflineAtariDataset(args.dataset, max_episodes=50)
    
    # Optimize data loading for multi-core CPUs
    num_workers = get_optimal_num_workers(device)
    use_persistent = num_workers > 0
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=(device=="cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None
    )
    print(f"Dataset size: {len(dataset)} transitions")
    print(f"DataLoader: {num_workers} workers, pin_memory={device=='cuda'}")
    
    # Get environment info
    sample_obs = dataset[0][0]
    obs_shape = sample_obs.shape
    num_actions = 4  # Breakout has 4 actions
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create encoder
    encoder = CNNEncoder(obs_shape, embed_dim=512)
    
    # Create agent based on method
    if args.method == "cql":
        # CQL agent
        cfg = AlgoConfig(
            device=device,
            lr_actor=1e-4,
            lr_critic=1e-4,
            lr_alpha=1e-4,
            lr_value=1e-4,
            gamma=0.99,
            tau=0.005,
            init_alpha=0.1,
            batch_size=args.batch_size,
            grad_norm_clip=10.0,
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        agent = CQLAgent(
            encoder=encoder,
            obs_shape=obs_shape,
            num_actions=num_actions,
            cfg=cfg,
            cql_alpha=args.cql_alpha,
            actor_hidden=256,
            critic_hidden=256
        )
        method_name = f"CQL (α={args.cql_alpha})"
    
    elif args.method == "iql":
        # IQL agent
        cfg = AlgoConfig(
            device=device,
            lr_actor=1e-4,
            lr_critic=1e-4,
            lr_alpha=1e-4,
            lr_value=1e-4,
            gamma=0.99,
            tau=0.005,
            init_alpha=0.1,
            batch_size=args.batch_size,
            grad_norm_clip=10.0,
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        agent = IQLAgent(
            encoder=encoder,
            obs_shape=obs_shape,
            num_actions=num_actions,
            cfg=cfg,
            expectile_tau=args.expectile_tau,
            beta_adv=args.beta_adv,
            actor_hidden=256,
            critic_hidden=256,
            value_hidden=256
        )
        method_name = f"IQL (τ={args.expectile_tau}, β={args.beta_adv})"
    
    elif args.method == "brac":
        # BRAC-BCQ agent
        cfg = AlgoConfig(
            device=device,
            lr_actor=1e-4,
            lr_critic=1e-4,
            lr_alpha=1e-4,
            lr_value=1e-4,  # For behavior policy
            gamma=0.99,
            tau=0.005,
            init_alpha=0.1,
            batch_size=args.batch_size,
            grad_norm_clip=10.0,
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        agent = BRACAgentDiscrete(
            obs_shape=obs_shape,
            num_actions=num_actions,
            embed_dim=512,
            hidden=256,
            cfg=cfg,
            lambda_kl=args.lambda_kl,
            entropy_coef=0.0,
            bcq_threshold=args.bcq_threshold
        )
        bcq_str = f", BCQ={args.bcq_threshold}" if args.bcq_threshold else ""
        method_name = f"BRAC (λ={args.lambda_kl}{bcq_str})"
    
    else:
        # ERSAC agent
        use_epinet = (args.method == "epinet")
        ersac_cfg = ERSACConfig(
            uncertainty_set=args.method,
            use_epinet=use_epinet,
            num_actions=num_actions,
            ensemble_size=5,
            gamma=0.99,
            tau=0.005,
            lr_critic=1e-4,
            lr_actor=1e-4,
            lr_alpha=1e-4,
            init_alpha=0.1,
            batch_size=args.batch_size,
            device=device,
            ellipsoid_coverage=0.9,
            min_variance=1e-6,
            epinet_z_dim=10,
            epinet_prior_scale=1.0,
            bootstrap_noise_scale=0.1,
            grad_norm_clip=10.0,
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        agent = ERSACAgentExtended(
            cfg=ersac_cfg,
            encoder=encoder,
            obs_shape=obs_shape
        )
        method_name = f"ERSAC-{args.method.upper()}"
    
    # Train agent
    agent = train_agent(agent, dataloader, args.steps, method_name)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
