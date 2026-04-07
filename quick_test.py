"""
Ultra-fast test script for quick verification
Uses minimal data and simplified training
Includes gradient flow checks and robustness tests
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime

from common_offline_discrete import CNNEncoder, AlgoConfig
from cql_discrete import CQLAgent
from iql_discrete import IQLAgent
from brac_bcq_discrete import BRACAgentDiscrete
from ersac_agent_extended import ERSACAgentExtended, ERSACConfig

def check_gradients(agent, batch, method_name):
    """
    Check that gradients are flowing correctly through all components.
    
    Returns:
        dict with gradient statistics
    """
    results = {
        'has_grad': True,
        'encoder_grad_norm': 0.0,
        'actor_grad_norm': 0.0,
        'critic_grad_norm': 0.0,
        'nan_grads': False,
        'inf_grads': False
    }
    
    # Zero all gradients
    if hasattr(agent, 'encoder'):
        agent.encoder.zero_grad()
    
    # Run update
    try:
        metrics = agent.update(batch)
    except Exception as e:
        results['has_grad'] = False
        results['error'] = str(e)
        return results
    
    # Check encoder gradients
    if hasattr(agent, 'encoder'):
        encoder_grad_norm = 0.0
        encoder_has_grad = False
        for param in agent.encoder.parameters():
            if param.grad is not None:
                encoder_has_grad = True
                encoder_grad_norm += param.grad.norm().item() ** 2
                if torch.isnan(param.grad).any():
                    results['nan_grads'] = True
                if torch.isinf(param.grad).any():
                    results['inf_grads'] = True
        results['encoder_grad_norm'] = encoder_grad_norm ** 0.5
        results['encoder_has_grad'] = encoder_has_grad
    
    # Check actor/policy gradients
    if hasattr(agent, 'policy'):
        actor_grad_norm = 0.0
        actor_has_grad = False
        for param in agent.policy.parameters():
            if param.grad is not None:
                actor_has_grad = True
                actor_grad_norm += param.grad.norm().item() ** 2
                if torch.isnan(param.grad).any():
                    results['nan_grads'] = True
                if torch.isinf(param.grad).any():
                    results['inf_grads'] = True
        results['actor_grad_norm'] = actor_grad_norm ** 0.5
        results['actor_has_grad'] = actor_has_grad
    elif hasattr(agent, 'actor'):
        actor_grad_norm = 0.0
        actor_has_grad = False
        for param in agent.actor.parameters():
            if param.grad is not None:
                actor_has_grad = True
                actor_grad_norm += param.grad.norm().item() ** 2
                if torch.isnan(param.grad).any():
                    results['nan_grads'] = True
                if torch.isinf(param.grad).any():
                    results['inf_grads'] = True
        results['actor_grad_norm'] = actor_grad_norm ** 0.5
        results['actor_has_grad'] = actor_has_grad
    
    # Check critic/Q gradients
    if hasattr(agent, 'q'):
        critic_grad_norm = 0.0
        critic_has_grad = False
        for param in agent.q.parameters():
            if param.grad is not None:
                critic_has_grad = True
                critic_grad_norm += param.grad.norm().item() ** 2
                if torch.isnan(param.grad).any():
                    results['nan_grads'] = True
                if torch.isinf(param.grad).any():
                    results['inf_grads'] = True
        results['critic_grad_norm'] = critic_grad_norm ** 0.5
        results['critic_has_grad'] = critic_has_grad
    elif hasattr(agent, 'q_net'):
        # Epinet case
        critic_grad_norm = 0.0
        critic_has_grad = False
        for param in agent.q_net.parameters():
            if param.grad is not None:
                critic_has_grad = True
                critic_grad_norm += param.grad.norm().item() ** 2
                if torch.isnan(param.grad).any():
                    results['nan_grads'] = True
                if torch.isinf(param.grad).any():
                    results['inf_grads'] = True
        results['critic_grad_norm'] = critic_grad_norm ** 0.5
        results['critic_has_grad'] = critic_has_grad
    elif hasattr(agent, 'q_nets'):
        # Ensemble case
        critic_grad_norm = 0.0
        critic_has_grad = False
        for q_net in agent.q_nets:
            for param in q_net.parameters():
                if param.grad is not None:
                    critic_has_grad = True
                    critic_grad_norm += param.grad.norm().item() ** 2
                    if torch.isnan(param.grad).any():
                        results['nan_grads'] = True
                    if torch.isinf(param.grad).any():
                        results['inf_grads'] = True
        results['critic_grad_norm'] = critic_grad_norm ** 0.5
        results['critic_has_grad'] = critic_has_grad
    
    return results


def test_robustness(agent, batch, method_name):
    """
    Test agent robustness to edge cases.
    
    Returns:
        dict with test results
    """
    results = {
        'handles_zero_rewards': False,
        'handles_extreme_states': False,
        'stable_over_steps': False,
        'no_nan_outputs': True,
        'no_inf_outputs': True
    }
    
    S, A, R, S2, D = batch
    
    # Test 1: Zero rewards
    try:
        batch_zero_r = (S, A, torch.zeros_like(R), S2, D)
        metrics = agent.update(batch_zero_r)
        if not (torch.isnan(torch.tensor(metrics['critic_loss'])) or 
                torch.isinf(torch.tensor(metrics['critic_loss']))):
            results['handles_zero_rewards'] = True
    except:
        pass
    
    # Test 2: Extreme state values (near boundaries)
    try:
        S_extreme = torch.clamp(S * 2.0, 0, 1)  # Push to boundaries
        batch_extreme = (S_extreme, A, R, S_extreme, D)
        metrics = agent.update(batch_extreme)
        if not (torch.isnan(torch.tensor(metrics['critic_loss'])) or 
                torch.isinf(torch.tensor(metrics['critic_loss']))):
            results['handles_extreme_states'] = True
    except:
        pass
    
    # Test 3: Multiple consecutive updates (stability)
    try:
        losses = []
        for _ in range(5):
            metrics = agent.update(batch)
            losses.append(metrics['critic_loss'])
        
        # Check no NaN/Inf
        for loss in losses:
            if np.isnan(loss) or np.isinf(loss):
                results['no_nan_outputs'] = False
                results['no_inf_outputs'] = False
                break
        else:
            results['stable_over_steps'] = True
    except:
        pass
    
    return results


print("=" * 70)
print("QUICK TEST - Minimal Training on Random Data")
print("WITH GRADIENT FLOW AND ROBUSTNESS CHECKS")
print("=" * 70)

# Create synthetic data (much faster than loading Minari)
batch_size = 32
num_batches = 10  # Only 10 batches per method
obs_shape = (12, 84, 84)  # Smaller resolution
num_actions = 4
device = "cpu"

print(f"\nGenerating synthetic data...")
print(f"  Batch size: {batch_size}")
print(f"  Num batches: {num_batches}")
print(f"  Obs shape: {obs_shape}")
print(f"  Num actions: {num_actions}")

# Generate random batches
batches = []
for _ in range(num_batches):
    S = torch.rand(batch_size, *obs_shape)
    A = torch.randint(0, num_actions, (batch_size,))
    R = torch.randn(batch_size) * 0.1
    S2 = torch.rand(batch_size, *obs_shape)
    D = torch.zeros(batch_size)
    batches.append((S, A, R, S2, D))

print(f"✓ Generated {num_batches} random batches")

# Test each method
methods_to_test = ["cql", "iql", "brac", "box", "convex_hull", "ellipsoidal", "epinet"]
results = []

for method in methods_to_test:
    print(f"\n{'='*70}")
    print(f"Testing: {method.upper()}")
    print(f"{'='*70}")
    
    try:
        # Create encoder
        encoder = CNNEncoder(obs_shape, embed_dim=256)  # Smaller embedding
        
        # Create agent
        if method == "cql":
            cfg = AlgoConfig(device=device, grad_norm_clip=5.0)
            agent = CQLAgent(encoder, obs_shape, num_actions, cfg, 
                           actor_hidden=128, critic_hidden=128)
        
        elif method == "iql":
            cfg = AlgoConfig(device=device, grad_norm_clip=5.0)
            agent = IQLAgent(encoder, obs_shape, num_actions, cfg,
                           actor_hidden=128, critic_hidden=128, value_hidden=128)
        
        elif method == "brac":
            cfg = AlgoConfig(device=device, grad_norm_clip=5.0)
            agent = BRACAgentDiscrete(obs_shape, num_actions, embed_dim=256,
                                    hidden=128, cfg=cfg)
        
        elif method in ["box", "convex_hull", "ellipsoidal", "epinet"]:
            use_epinet = (method == "epinet")
            ersac_cfg = ERSACConfig(
                uncertainty_set=method,
                use_epinet=use_epinet,
                num_actions=num_actions,
                ensemble_size=3,  # Smaller ensemble for speed
                lr_critic=1e-3,
                lr_actor=1e-3
            )
            agent = ERSACAgentExtended(ersac_cfg, obs_shape, num_actions, 
                                      encoder, device)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ Agent created")
        
        # Gradient flow check
        print(f"  Checking gradient flow...")
        grad_results = check_gradients(agent, batches[0], method)
        
        if grad_results['nan_grads']:
            print(f"  ⚠ WARNING: NaN gradients detected!")
        if grad_results['inf_grads']:
            print(f"  ⚠ WARNING: Inf gradients detected!")
        
        if grad_results.get('encoder_has_grad', False):
            print(f"  ✓ Encoder gradients: {grad_results['encoder_grad_norm']:.4f}")
        if grad_results.get('actor_has_grad', False):
            print(f"  ✓ Actor gradients: {grad_results['actor_grad_norm']:.4f}")
        if grad_results.get('critic_has_grad', False):
            print(f"  ✓ Critic gradients: {grad_results['critic_grad_norm']:.4f}")
        
        # Robustness tests
        print(f"  Running robustness tests...")
        robust_results = test_robustness(agent, batches[0], method)
        
        passed_tests = sum([
            robust_results['handles_zero_rewards'],
            robust_results['handles_extreme_states'],
            robust_results['stable_over_steps'],
            robust_results['no_nan_outputs'],
            robust_results['no_inf_outputs']
        ])
        print(f"  ✓ Robustness: {passed_tests}/5 tests passed")
        
        if not robust_results['no_nan_outputs']:
            print(f"  ⚠ WARNING: NaN outputs detected in robustness test!")
        if not robust_results['no_inf_outputs']:
            print(f"  ⚠ WARNING: Inf outputs detected in robustness test!")
        
        # Quick training
        start = time.time()
        for i, batch in enumerate(batches):
            metrics = agent.update(batch)
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1}/{num_batches} | "
                      f"Critic Loss: {metrics.get('critic_loss', 0):.3f}")
        
        elapsed = time.time() - start
        print(f"✓ Training complete in {elapsed:.2f}s")
        
        # Store results with additional metrics
        results.append((
            method, 
            "SUCCESS", 
            elapsed,
            grad_results,
            robust_results
        ))
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append((method, f"FAILED: {e}", 0, {}, {}))

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Method':<15} {'Status':<15} {'Time (s)':<10} {'Gradients':<12} {'Robust':<8}")
print("-" * 70)

for item in results:
    if len(item) >= 5:
        method, status, elapsed, grad_res, robust_res = item[:5]
        
        # Gradient status
        if grad_res:
            grad_ok = (not grad_res.get('nan_grads', True) and 
                      not grad_res.get('inf_grads', True))
            grad_status = "✓" if grad_ok else "⚠"
        else:
            grad_status = "N/A"
        
        # Robustness status
        if robust_res:
            robust_score = sum([
                robust_res.get('handles_zero_rewards', False),
                robust_res.get('handles_extreme_states', False),
                robust_res.get('stable_over_steps', False),
                robust_res.get('no_nan_outputs', False),
                robust_res.get('no_inf_outputs', False)
            ])
            robust_status = f"{robust_score}/5"
        else:
            robust_status = "N/A"
        
        status_short = "SUCCESS" if "SUCCESS" in status else "FAILED"
        print(f"{method:<15} {status_short:<15} {elapsed:<10.2f} {grad_status:<12} {robust_status:<8}")
    else:
        # Old format compatibility
        method, status, elapsed = item[:3]
        status_short = "SUCCESS" if "SUCCESS" in status else "FAILED"
        print(f"{method:<15} {status_short:<15} {elapsed:<10.2f} {'N/A':<12} {'N/A':<8}")

print()
print("Legend:")
print("  Gradients: ✓ = No NaN/Inf, ⚠ = Issues detected")
print("  Robust: X/5 = Passed X out of 5 robustness tests")

# Check for warnings
warnings = []
for item in results:
    if len(item) >= 5:
        method, status, elapsed, grad_res, robust_res = item[:5]
        if grad_res and (grad_res.get('nan_grads') or grad_res.get('inf_grads')):
            warnings.append(f"  ⚠ {method}: Gradient issues detected")
        if robust_res and not robust_res.get('no_nan_outputs'):
            warnings.append(f"  ⚠ {method}: NaN outputs in robustness test")
        if robust_res and not robust_res.get('stable_over_steps'):
            warnings.append(f"  ⚠ {method}: Instability over consecutive updates")

if warnings:
    print(f"\n{'='*70}")
    print("WARNINGS:")
    print(f"{'='*70}")
    for w in warnings:
        print(w)

print(f"\n✓ Quick test complete!")
print("If all methods show SUCCESS with ✓ gradients and high robustness scores,")
print("your setup is working correctly.")
print("You can now run the full experiment with:")
print("  python run_all_experiments.py --steps 5000 --max_episodes 10")

# Additional note about Epinet
print("\nNote: Epinet shows similar or slightly slower training times because:")
print("  - Training requires multiple samples for uncertainty estimation")
print("  - More complex network architecture (epistemic index concatenation)")
print("  - Covariance computation for ellipsoidal uncertainty set")
print("Epinet's advantage is during INFERENCE: single forward pass vs N ensemble members")

