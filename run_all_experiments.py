# run_all_experiments.py
"""
Run all offline RL methods on a given environment and save results
Trains each method and evaluates on test data with comprehensive metrics
"""

import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import minari
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
    """Dataset wrapper for Minari Atari datasets with train/test split"""
    
    def __init__(self, dataset_name: str, max_episodes: int = None, 
                 train_ratio: float = 0.8, split: str = "train"):
        self.dataset = minari.load_dataset(dataset_name)
        
        # Get total number of episodes
        total_episodes = self.dataset.total_episodes
        if max_episodes is not None:
            total_episodes = min(total_episodes, max_episodes)
        
        print(f"Loading {total_episodes} episodes for {split} split...")
        
        # Iterate through episodes
        all_episodes = []
        for ep_idx, episode in enumerate(self.dataset.iterate_episodes()):
            if ep_idx >= total_episodes:
                break
            all_episodes.append(episode)
            if (ep_idx + 1) % 5 == 0:
                print(f"  Loaded {ep_idx + 1}/{total_episodes} episodes...", end='\r')
        
        print(f"  Loaded {len(all_episodes)} episodes" + " " * 20)
        
        # Split into train/test
        n_train = int(len(all_episodes) * train_ratio)
        if split == "train":
            self.episodes = all_episodes[:n_train]
        elif split == "test":
            self.episodes = all_episodes[n_train:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
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
        """Build dataset from episodes (optimized)"""
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones = [], []
        
        print(f"  Building {len(self.episodes)} episodes into transitions...")
        
        for ep_idx, ep in enumerate(self.episodes):
            # Pre-process all observations for this episode at once
            obs_list = [self._preprocess_obs(o) for o in ep.observations]
            
            # Pre-compute all frame stacks for this episode
            stacked_frames = []
            k = 4
            for t in range(len(obs_list)):
                if t < k - 1:
                    # Pad with first frame
                    frames = [obs_list[0]] * (k - 1 - t) + obs_list[:t+1]
                else:
                    frames = obs_list[t-k+1:t+1]
                stacked = np.concatenate(frames, axis=0)
                stacked_frames.append(stacked)
            
            # Now create transitions
            for t in range(len(ep.actions)):
                s = stacked_frames[t]
                s2 = stacked_frames[min(t+1, len(stacked_frames)-1)]
                
                # Clip rewards to [-1, 1]
                r = np.clip(ep.rewards[t], -1.0, 1.0)
                
                self.states.append(s)
                self.actions.append(ep.actions[t])
                self.rewards.append(r)
                self.next_states.append(s2)
                self.dones.append(ep.terminations[t] or ep.truncations[t])
            
            if (ep_idx + 1) % 5 == 0 or (ep_idx + 1) == len(self.episodes):
                print(f"    Processed {ep_idx + 1}/{len(self.episodes)} episodes...", end='\r')
        
        print(f"    Processed {len(self.episodes)} episodes - {len(self.states)} transitions" + " " * 20)
        
        # Pre-convert to tensors for faster training
        print("  Converting to tensors...")
        self.states = torch.from_numpy(np.stack(self.states, axis=0))
        self.actions = torch.tensor(self.actions, dtype=torch.long)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        self.next_states = torch.from_numpy(np.stack(self.next_states, axis=0))
        self.dones = torch.tensor(self.dones, dtype=torch.float32)
        print("  ✓ Dataset ready!")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )


def create_agent(method: str, encoder: CNNEncoder, obs_shape, num_actions: int, 
                 device: str, config: Dict):
    """Create agent based on method name"""
    
    if method == "cql":
        cfg = AlgoConfig(
            device=device,
            lr_actor=config.get("lr_actor", 1e-4),
            lr_critic=config.get("lr_critic", 1e-4),
            lr_alpha=config.get("lr_alpha", 1e-4),
            lr_value=config.get("lr_value", 1e-4),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
            batch_size=config.get("batch_size", 256),
            grad_norm_clip=config.get("grad_norm_clip", 10.0),
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        return CQLAgent(
            encoder=encoder,
            obs_shape=obs_shape,
            num_actions=num_actions,
            cfg=cfg,
            cql_alpha=config.get("cql_alpha", 1.0),
            actor_hidden=256,
            critic_hidden=256
        )
    
    elif method == "iql":
        cfg = AlgoConfig(
            device=device,
            lr_actor=config.get("lr_actor", 1e-4),
            lr_critic=config.get("lr_critic", 1e-4),
            lr_value=config.get("lr_value", 1e-4),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
            batch_size=config.get("batch_size", 256),
            grad_norm_clip=config.get("grad_norm_clip", 10.0),
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        return IQLAgent(
            encoder=encoder,
            obs_shape=obs_shape,
            num_actions=num_actions,
            cfg=cfg,
            expectile_tau=config.get("expectile_tau", 0.7),
            beta_adv=config.get("beta_adv", 3.0),
            actor_hidden=256,
            critic_hidden=256,
            value_hidden=256
        )
    
    elif method == "brac":
        cfg = AlgoConfig(
            device=device,
            lr_actor=config.get("lr_actor", 1e-4),
            lr_critic=config.get("lr_critic", 1e-4),
            lr_value=config.get("lr_value", 1e-4),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
            batch_size=config.get("batch_size", 256),
            grad_norm_clip=config.get("grad_norm_clip", 10.0),
            clamp_value_min=-50.0,
            clamp_value_max=50.0
        )
        return BRACAgentDiscrete(
            obs_shape=obs_shape,
            num_actions=num_actions,
            embed_dim=512,
            hidden=256,
            cfg=cfg,
            lambda_kl=config.get("lambda_kl", 1.0),
            entropy_coef=0.0,
            bcq_threshold=config.get("bcq_threshold", None)
        )
    
    elif method in ["box", "convex_hull", "ellipsoidal", "epinet"]:
        use_epinet = (method == "epinet")
        ersac_cfg = ERSACConfig(
            uncertainty_set=method,
            use_epinet=use_epinet,
            num_actions=num_actions,
            ensemble_size=config.get("ensemble_size", 5),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
            lr_critic=config.get("lr_critic", 1e-4),
            lr_actor=config.get("lr_actor", 1e-4),
            lr_alpha=config.get("lr_alpha", 1e-4),
            init_alpha=config.get("init_alpha", 0.1),
            learn_alpha=config.get("learn_alpha", True),
            ellipsoid_coverage=config.get("ellipsoid_coverage", 0.9),
            min_variance=1e-6,
            epinet_z_dim=config.get("epinet_z_dim", 10),
            epinet_prior_scale=1.0,
            bootstrap_noise_scale=config.get("bootstrap_noise_scale", 0.1),
            bc_weight=config.get("bc_weight", 1.0),  # BC weight for discrete offline RL
            grad_norm_clip=config.get("grad_norm_clip", 10.0),
            edac_eta=config.get("edac_eta", 0.0)  # EDAC diversity regularization
        )
        return ERSACAgentExtended(
            cfg=ersac_cfg,
            obs_shape=obs_shape,
            num_actions=num_actions,
            encoder=encoder,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def evaluate_agent(agent, test_loader: DataLoader, device: str) -> Dict:
    """Evaluate agent on test set"""
    
    # Set to eval mode
    agent.encoder.eval()
    if hasattr(agent, 'actor'):
        agent.actor.eval()
    
    total_loss = 0.0
    total_q_value = 0.0
    correct_actions = 0
    total_samples = 0
    
    all_q_values = []
    all_rewards = []
    
    with torch.no_grad():
        for batch in test_loader:
            S, A, R, S2, D = [b.to(device) for b in batch]
            A = A.long()
            
            # Get Q-values and greedy actions
            if not hasattr(agent, 'encoder'):
                continue
                
            z = agent.encoder(S)
            
            # For ERSAC agents, use the actor's greedy policy
            if hasattr(agent, 'actor') and hasattr(agent, 'q_nets'):
                # ERSAC: use actor for greedy action selection
                pi, _, _ = agent.actor(z)
                greedy_actions = pi.argmax(-1)
                # Still get Q-values for logging
                q_list = [q_net(z) for q_net in agent.q_nets]
                q_values = torch.stack(q_list, dim=0).mean(0)
            elif hasattr(agent, 'actor') and hasattr(agent, 'q_net'):
                # ERSAC Epinet: use actor for greedy action selection
                pi, _, _ = agent.actor(z)
                greedy_actions = pi.argmax(-1)
                # Get mean Q-values (forward with z=None returns mean)
                q_values = agent.q_net(z)  # z=None default returns mu
            elif hasattr(agent, 'q'):
                q_values = agent.q(z)
                greedy_actions = q_values.argmax(-1)
            elif hasattr(agent, 'q_nets'):
                # Non-ERSAC agent with q_nets
                q_list = [q_net(z) for q_net in agent.q_nets]
                q_values = torch.stack(q_list, dim=0).mean(0)
                greedy_actions = q_values.argmax(-1)
            elif hasattr(agent, 'critic_ensemble'):
                # Legacy ERSAC agent with critic_ensemble
                q_list = [q_net(z) for q_net in agent.critic_ensemble]
                q_values = torch.stack(q_list, dim=0).mean(0)
                greedy_actions = q_values.argmax(-1)
            else:
                continue
            
            # Get Q-value for taken action
            q_a = q_values.gather(1, A.unsqueeze(1)).squeeze(1)
            
            # Check if greedy action matches expert action
            correct_actions += (greedy_actions == A).sum().item()
            
            # Accumulate stats
            total_q_value += q_a.sum().item()
            total_samples += len(A)
            all_q_values.extend(q_a.cpu().numpy().tolist())
            all_rewards.extend(R.cpu().numpy().tolist())
    
    # Restore train mode
    agent.encoder.train()
    if hasattr(agent, 'actor'):
        agent.actor.train()
    
    return {
        "mean_q_value": total_q_value / total_samples if total_samples > 0 else 0.0,
        "action_agreement": correct_actions / total_samples if total_samples > 0 else 0.0,
        "mean_reward": np.mean(all_rewards),
        "std_q_value": np.std(all_q_values),
        "num_samples": total_samples
    }


def train_and_evaluate(method: str, train_loader: DataLoader, test_loader: DataLoader,
                       obs_shape, num_actions: int, device: str, 
                       num_steps: int, eval_freq: int, config: Dict,
                       save_dir: Path,
                       early_stopping: bool = True,
                       patience: int = 5,
                       min_delta: float = 0.005) -> Dict:
    """
    Train a single method and evaluate periodically.
    
    Stopping criteria (aligned with D4RL/CQL/IQL literature):
    1. Fixed budget: Maximum `num_steps` gradient updates
    2. Early stopping: Stop if validation metric doesn't improve for `patience` evaluations
    3. Best checkpoint: Save model with best action_agreement on held-out data
    
    Args:
        early_stopping: Enable early stopping based on validation metrics
        patience: Number of evaluations without improvement before stopping
        min_delta: Minimum improvement to count as "better"
    """
    
    print(f"\n{'='*70}")
    print(f"Training: {method.upper()}")
    print(f"{'='*70}")
    
    try:
        # Create agent
        encoder = CNNEncoder(obs_shape, embed_dim=512)
        agent = create_agent(method, encoder, obs_shape, num_actions, device, config)
        print(f"✓ Agent created successfully")
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Training loop
    train_iter = iter(train_loader)
    training_metrics = []
    eval_metrics = []
    
    # Early stopping state (following IQL/CQL evaluation protocol)
    best_action_agreement = 0.0
    best_step = 0
    best_model_state = None
    patience_counter = 0
    stopped_early = False
    
    start_time = time.time()
    
    for step in range(1, num_steps + 1):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Update agent
        try:
            metrics = agent.update(batch)
            training_metrics.append(metrics)
        except Exception as e:
            print(f"✗ Update failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Progress indicator (every 50 steps)
        if step % 50 == 0 and step % eval_freq != 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta = (num_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            
            # Build detailed metrics string - Line 1: Losses
            metric_str = f"Step {step:6d}/{num_steps}"
            if 'td_loss' in metrics:
                metric_str += f" | TD: {metrics['td_loss']:6.4f}"
            if 'cql_pen' in metrics:
                metric_str += f" | CQL: {metrics['cql_pen']:6.4f}"
            if 'q_reg' in metrics:
                metric_str += f" | Qreg: {metrics['q_reg']:6.4f}"
            if 'value_loss' in metrics:
                metric_str += f" | V: {metrics['value_loss']:6.4f}"
            metric_str += f" | Critic: {metrics.get('critic_loss', 0):6.4f}"
            if 'actor_loss' in metrics:
                metric_str += f" | Actor: {metrics['actor_loss']:7.4f}"
            # BRAC-specific: behavior policy loss and KL
            if 'mu_ce' in metrics:
                metric_str += f" | μCE: {metrics['mu_ce']:6.4f}"
            if 'kl' in metrics:
                metric_str += f" | KL: {metrics['kl']:6.4f}"
            
            # Line 2: Q-values and other metrics
            q_str = "       "
            if 'mean_q' in metrics:
                q_str += f" | MeanQ: {metrics['mean_q']:6.2f}"
            if 'q_data' in metrics:
                q_str += f" | Q(data): {metrics['q_data']:6.2f}"
            if 'mean_v' in metrics:
                q_str += f" | V: {metrics['mean_v']:6.2f}"
            if 'adv_data' in metrics:
                q_str += f" | Adv: {metrics['adv_data']:6.2f}"
            if 'action_agree' in metrics:
                q_str += f" | ActAgree: {metrics['action_agree']:.3f}"
            if 'bc_loss' in metrics:
                q_str += f" | BC: {metrics['bc_loss']:.3f}"
            if 'mu_acc' in metrics:
                q_str += f" | μAcc: {metrics['mu_acc']:.3f}"
            if 'q_std' in metrics:
                q_str += f" | QStd: {metrics['q_std']:.3f}"
            if 'min_q_std' in metrics:
                q_str += f" | minQstd: {metrics['min_q_std']:.3f}"
            if 'alpha' in metrics:
                q_str += f" | α: {metrics['alpha']:.4f}"
            q_str += f" | {steps_per_sec:.1f} steps/s | ETA: {eta/60:.1f}min"
            
            # Line 3: Gradient norms (for debugging)
            grad_str = ""
            if 'critic_grad' in metrics:
                grad_str = f"        Grads: Critic={metrics['critic_grad']:.3f}"
                if 'encoder_grad_c' in metrics:
                    grad_str += f" | Enc(C)={metrics['encoder_grad_c']:.3f}"
                if 'actor_grad' in metrics:
                    grad_str += f" | Actor={metrics['actor_grad']:.3f}"
                if 'encoder_grad_a' in metrics:
                    grad_str += f" | Enc(A)={metrics['encoder_grad_a']:.3f}"
            
            print(metric_str)
            print(q_str)
            if grad_str:
                print(grad_str)
        
        # Periodic evaluation on held-out test set
        if step % eval_freq == 0:
            eval_result = evaluate_agent(agent, test_loader, device)
            eval_result['step'] = step
            eval_metrics.append(eval_result)
            
            current_metric = eval_result['action_agreement']
            
            # Best model checkpointing (D4RL standard practice)
            if current_metric > best_action_agreement + min_delta:
                best_action_agreement = current_metric
                best_step = step
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    'encoder': {k: v.cpu().clone() for k, v in agent.encoder.state_dict().items()},
                    'step': step,
                    'action_agreement': current_metric
                }
                improvement_marker = "★ NEW BEST"
            else:
                patience_counter += 1
                improvement_marker = f"(patience: {patience_counter}/{patience})"
            
            # Log progress
            elapsed = time.time() - start_time
            print(f"\n{'─'*70}")
            print(f"EVAL Step {step:6d}/{num_steps} | "
                  f"Test Action Agreement: {eval_result['action_agreement']:.3f} | "
                  f"Test Q: {eval_result['mean_q_value']:6.2f} | "
                  f"{improvement_marker}")
            print(f"     Best so far: {best_action_agreement:.3f} at step {best_step} | "
                  f"Time: {elapsed/60:.1f}min")
            print(f"{'─'*70}\n")
            
            # Early stopping check (following IQL paper methodology)
            if early_stopping and patience_counter >= patience:
                print(f"\n{'='*70}")
                print(f"EARLY STOPPING: No improvement for {patience} evaluations")
                print(f"Best action agreement: {best_action_agreement:.3f} at step {best_step}")
                print(f"{'='*70}\n")
                stopped_early = True
                break
    
    train_time = time.time() - start_time
    actual_steps = step  # Record actual steps taken
    
    # Final evaluation (on last model)
    final_eval = evaluate_agent(agent, test_loader, device)
    final_eval['step'] = actual_steps
    
    # Determine which model to save (best vs final)
    # Following D4RL convention: save BEST model by validation metric
    if best_model_state is not None and best_action_agreement > final_eval['action_agreement']:
        # Restore best model for saving
        agent.encoder.load_state_dict({k: v.to(device) for k, v in best_model_state['encoder'].items()})
        use_best = True
        save_eval = {'action_agreement': best_action_agreement, 'step': best_step, 
                     'mean_q_value': final_eval['mean_q_value']}  # Q-value from final for reference
    else:
        use_best = False
        save_eval = final_eval
    
    # Save model (best checkpoint - standard in D4RL/IQL)
    model_path = save_dir / f"{method}_best.pt"
    torch.save({
        'agent_state': agent.state_dict() if hasattr(agent, 'state_dict') else None,
        'encoder_state': agent.encoder.state_dict() if hasattr(agent, 'encoder') else None,
        'config': config,
        'final_eval': save_eval,
        'best_step': best_step,
        'best_action_agreement': best_action_agreement,
        'stopped_early': stopped_early,
        'actual_steps': actual_steps
    }, model_path)
    
    print(f"\n{'='*70}")
    print(f"{method.upper()} TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Stopping: {'Early (patience={})'.format(patience) if stopped_early else 'Completed all steps'}")
    print(f"  Actual steps: {actual_steps}/{num_steps}")
    print(f"  Best action agreement: {best_action_agreement:.3f} (step {best_step})")
    print(f"  Final action agreement: {final_eval['action_agreement']:.3f}")
    print(f"  Model saved: {'BEST checkpoint' if use_best else 'Final checkpoint'}")
    print(f"  Training Time: {train_time:.1f}s ({train_time/60:.1f}min)")
    print(f"  Model path: {model_path}")
    print(f"{'='*70}\n")
    
    return {
        'method': method,
        'training_metrics': training_metrics,
        'eval_metrics': eval_metrics,
        'final_eval': final_eval,
        'best_eval': {'action_agreement': best_action_agreement, 'step': best_step},
        'train_time': train_time,
        'actual_steps': actual_steps,
        'stopped_early': stopped_early,
        'model_path': str(model_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Run all offline RL experiments")
    parser.add_argument("--dataset", type=str, default="atari/breakout/expert-v0",
                       help="Minari dataset name")
    parser.add_argument("--methods", type=str, nargs="+", 
                       default=["cql", "iql", "brac", "box", "convex_hull", "ellipsoidal", "epinet"],
                       help="Methods to run (default: all)")
    parser.add_argument("--steps", type=int, default=50000,
                       help="Maximum training steps per method")
    parser.add_argument("--eval_freq", type=int, default=5000,
                       help="Evaluation frequency (steps)")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--max_episodes", type=int, default=50,
                       help="Max episodes to load from dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Train/test split ratio")
    parser.add_argument("--output_dir", type=str, default="./experiment_results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Early stopping arguments (following IQL/CQL literature)
    parser.add_argument("--early_stopping", action="store_true", default=True,
                       help="Enable early stopping (default: True)")
    parser.add_argument("--no_early_stopping", action="store_false", dest="early_stopping",
                       help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience (number of evals without improvement)")
    parser.add_argument("--min_delta", type=float, default=0.005,
                       help="Minimum improvement to reset patience")
    
    # ERSAC-specific arguments
    parser.add_argument("--ensemble_size", type=int, default=10,
                       help="Number of Q-networks in ensemble (ERSAC paper uses 10)")
    parser.add_argument("--bc_weight", type=float, default=1.0,
                       help="BC loss weight for discrete ERSAC (higher = more imitation)")
    parser.add_argument("--init_alpha", type=float, default=0.1,
                       help="Initial entropy coefficient (alpha)")
    parser.add_argument("--no_learn_alpha", action="store_true",
                       help="Fix alpha at init_alpha (don't learn it)")
    parser.add_argument("--bootstrap_noise", type=float, default=0.1,
                       help="Bootstrap noise scale for ensemble diversity (try 1.0 for more diversity)")
    parser.add_argument("--edac_eta", type=float, default=0.0,
                       help="EDAC diversity regularization (0=off, try 1.0-10.0 for diversity)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset.replace("/", "_")
    save_dir = Path(args.output_dir) / f"{dataset_name}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {save_dir}")
    
    # Load datasets
    print(f"\nLoading dataset: {args.dataset}")
    train_dataset = OfflineAtariDataset(args.dataset, max_episodes=args.max_episodes, 
                                       train_ratio=args.train_ratio, split="train")
    test_dataset = OfflineAtariDataset(args.dataset, max_episodes=args.max_episodes,
                                      train_ratio=args.train_ratio, split="test")
    
    # Optimize data loading for multi-core CPUs
    num_workers = get_optimal_num_workers(device)
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=(device=="cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=(device=="cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None
    )
    
    print(f"DataLoader: {num_workers} workers, pin_memory={device=='cuda'}")
    
    print(f"Train size: {len(train_dataset)} transitions")
    print(f"Test size: {len(test_dataset)} transitions")
    
    # Get environment info
    sample_obs = train_dataset[0][0]
    obs_shape = sample_obs.shape
    
    # Get number of actions from the dataset's action space
    try:
        num_actions = train_dataset.dataset.action_space.n
    except:
        # Fallback: check max action in dataset
        all_actions = []
        for i in range(min(100, len(train_dataset))):
            all_actions.append(train_dataset[i][1].item())
        num_actions = int(max(all_actions)) + 1
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Default config for all methods
    # Learning rates follow offline RL conventions:
    # - Critic LR higher (learns Q-values first)
    # - Actor LR lower (follows critic, prevents overfitting to noisy Q-estimates)
    # - Ratio ~3:1 to 10:1 is common (CQL paper uses 3:1, IQL uses 10:1)
    default_config = {
        "lr_actor": 3e-5,   # Lower for stability (IQL-style)
        "lr_critic": 3e-4,  # Higher for faster Q-learning
        "lr_alpha": 3e-5,   # Match actor LR
        "lr_value": 3e-4,   # Same as critic (for IQL)
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": args.batch_size,
        "grad_norm_clip": 10.0,
        # Method-specific defaults
        "cql_alpha": 1.0,
        "expectile_tau": 0.7,
        "beta_adv": 3.0,
        "lambda_kl": 1.0,
        "bcq_threshold": 0.3,  # Enable BCQ filtering for discrete BRAC (standard adaptation)
        "ensemble_size": args.ensemble_size,  # ERSAC paper uses N=10
        "bc_weight": args.bc_weight,  # BC weight for discrete offline RL
        "init_alpha": args.init_alpha,
        "learn_alpha": not args.no_learn_alpha,
        "bootstrap_noise_scale": args.bootstrap_noise,  # For ensemble diversity
        "edac_eta": args.edac_eta,  # EDAC diversity regularization
        "ellipsoid_coverage": 0.9,
        "epinet_z_dim": 10
    }
    
    # Run all experiments
    all_results = []
    
    print(f"\n{'='*70}")
    print(f"STOPPING CRITERIA (Literature-aligned)")
    print(f"{'='*70}")
    print(f"  Max steps: {args.steps}")
    print(f"  Early stopping: {args.early_stopping}")
    if args.early_stopping:
        print(f"  Patience: {args.patience} evaluations")
        print(f"  Min improvement: {args.min_delta}")
    print(f"  Metric: Action Agreement on held-out test set")
    print(f"  Model selection: Best checkpoint (D4RL standard)")
    print(f"{'='*70}\n")
    
    for method in args.methods:
        try:
            result = train_and_evaluate(
                method=method,
                train_loader=train_loader,
                test_loader=test_loader,
                obs_shape=obs_shape,
                num_actions=num_actions,
                device=device,
                num_steps=args.steps,
                eval_freq=args.eval_freq,
                config=default_config,
                save_dir=save_dir,
                early_stopping=args.early_stopping,
                patience=args.patience,
                min_delta=args.min_delta
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error training {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary results
    summary = {
        'dataset': args.dataset,
        'num_steps': args.steps,
        'seed': args.seed,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'methods': args.methods,
        'config': default_config,
        'results': []
    }
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'Test Q':<12} {'Action Agree':<15} {'Time (s)':<10}")
    print("-" * 70)
    
    for result in all_results:
        method = result['method']
        final = result['final_eval']
        train_time = result['train_time']
        
        print(f"{method:<15} {final['mean_q_value']:<12.4f} "
              f"{final['action_agreement']:<15.3f} {train_time:<10.1f}")
        
        summary['results'].append({
            'method': method,
            'final_test_q': final['mean_q_value'],
            'action_agreement': final['action_agreement'],
            'train_time': train_time,
            'model_path': result['model_path']
        })
    
    # Save summary JSON
    summary_path = save_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    results_path = save_dir / "detailed_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON
        json_results = []
        for r in all_results:
            json_r = {
                'method': r['method'],
                'train_time': r['train_time'],
                'model_path': r['model_path'],
                'final_eval': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                              for k, v in r['final_eval'].items()},
                'eval_history': r['eval_metrics']
            }
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ All experiments complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
