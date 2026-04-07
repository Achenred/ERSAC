"""Quick test to verify all agents can be created"""
import torch
from common_offline_discrete import CNNEncoder, AlgoConfig
from cql_discrete import CQLAgent
from iql_discrete import IQLAgent
from brac_bcq_discrete import BRACAgentDiscrete
from ersac_agent_extended import ERSACAgentExtended, ERSACConfig

# Test setup
obs_shape = (12, 210, 160)  # 4 frames stacked, Atari resolution
num_actions = 4
device = "cpu"

print("Testing agent creation...")
print("=" * 60)

# Create encoder
encoder = CNNEncoder(obs_shape, embed_dim=512)
print(f"✓ Encoder created (embed_dim={encoder.embed_dim})")

# Test CQL
try:
    cfg = AlgoConfig(device=device)
    agent = CQLAgent(encoder, obs_shape, num_actions, cfg)
    print("✓ CQL agent created successfully")
except Exception as e:
    print(f"✗ CQL failed: {e}")

# Test IQL
try:
    cfg = AlgoConfig(device=device)
    agent = IQLAgent(encoder, obs_shape, num_actions, cfg)
    print("✓ IQL agent created successfully")
except Exception as e:
    print(f"✗ IQL failed: {e}")

# Test BRAC
try:
    cfg = AlgoConfig(device=device)
    agent = BRACAgentDiscrete(obs_shape, num_actions, embed_dim=512, hidden=256, cfg=cfg)
    print("✓ BRAC agent created successfully")
except Exception as e:
    print(f"✗ BRAC failed: {e}")

# Test ERSAC variants
for method in ["box", "convex_hull", "ellipsoidal", "epinet"]:
    try:
        use_epinet = (method == "epinet")
        cfg = ERSACConfig(
            uncertainty_set=method,
            use_epinet=use_epinet,
            num_actions=num_actions,
            ensemble_size=5
        )
        agent = ERSACAgentExtended(cfg, obs_shape, num_actions, encoder, device)
        print(f"✓ ERSAC-{method.upper()} agent created successfully")
    except Exception as e:
        print(f"✗ ERSAC-{method} failed: {e}")

print("=" * 60)
print("All tests complete!")
