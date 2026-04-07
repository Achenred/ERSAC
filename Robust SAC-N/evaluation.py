import numpy as np
from tabulate import tabulate
import os
import torch
import gym
from collections import defaultdict
import pandas as pd

from tabulate import tabulate
from operator import itemgetter

from train_new_lander import Actor as learned_actor
from data_generation_dynamic import Actor as opt_actor


def load_policy(path, actor_class, state_dim, action_dim, device="cpu"):

    actor = actor_class(state_dim, action_dim).to(device)

    actor.load_state_dict(torch.load(path, map_location=device))
    actor.eval()
    return actor

def evaluate_policy(actor, env, num_episodes=10, max_steps=1000):
    actor.eval()
    try:
        device = next(actor.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    returns = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                output = actor(state_tensor)

            if isinstance(output, torch.distributions.Categorical):
                action = output.sample().item()
            else:
                action = torch.argmax(output, dim=1).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        returns.append(total_reward)

    returns = np.array(returns)
    top_10 = np.percentile(returns, 50)
    mean_top_10 = returns[returns >= top_10].mean()

    return mean_top_10


def parse_filename(fname):
    """
    Parses filename like 'LunarLander-v2_box_data_size_10000_tau_0.9_actor.pt'
    """
    parts = fname.replace("_actor.pt", "").split("_")

    try:
        env_name = parts[0]  # e.g. LunarLander-v2
        model_type = parts[1]
        data_size = None
        tau = None
        for i, part in enumerate(parts):
            if part == "size":
                data_size = int(parts[i + 1])
            if part == "tau":
                tau = float(parts[i + 1])
        return env_name, model_type, data_size, tau
    except Exception as e:
        print(f"Failed to parse filename: {fname} ({e})")
        return None, None, None, None



def evaluate_all_models(trained_dir, models_dir, learned_actor_class, opt_actor_class, env_name, num_episodes=10, device="cpu"):
    results = []

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n


    # Define and evaluate random policy once
    class RandomPolicy(torch.nn.Module):
        def forward(self, x):
            return torch.ones((x.shape[0], action_dim)) / action_dim
    random_actor = RandomPolicy()
    V_random = evaluate_policy(random_actor, env, num_episodes)

    # Load and evaluate benchmark optimal policy (tau=0.5)
    benchmark_tau = "0.5"
    benchmark_path = os.path.join(models_dir, f"{env_name}/best_actor_tau_{benchmark_tau}.pt")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark optimal model not found at {benchmark_path}")
    benchmark_actor = load_policy(benchmark_path, opt_actor_class, state_dim, action_dim, device)
    V_opt_benchmark = evaluate_policy(benchmark_actor, env, num_episodes)

    # === Evaluate learned models ===
    for fname in os.listdir(trained_dir):

        if not fname.endswith("_actor.pt"):
            continue

        env_tag, model_type, data_size, tau = parse_filename(fname)
        if env_tag not in env_name:
            continue
        if None in (env_tag, model_type, data_size, tau):
            print(f"Skipping malformed filename: {fname}")
            continue
        if model_type in ("epinet", "ellipsoid", "epinet_nl"):
            continue

        model_path = os.path.join(trained_dir, fname)
        actor = load_policy(model_path, learned_actor_class, state_dim, action_dim, device)
        V_model = evaluate_policy(actor, env, num_episodes)

        normalized_improvement = (V_model - V_random) / (V_opt_benchmark - V_random + 1e-8)

        results.append({
            "env": env_tag,
            "data_size": int(data_size),
            "tau": float(tau),
            "model": model_type,
            "V_model": round(V_model, 2),
            "V_opt_tau_0.5": round(V_opt_benchmark, 2),
            "V_random": round(V_random, 2),
            "normalized_improvement": round(normalized_improvement, 4)
        })

    # === Evaluate all optimal models (tau ≠ 0.5) ===
    for fname in os.listdir(os.path.join(models_dir, env_name)):
        if not fname.startswith("best_actor_tau_") or not fname.endswith(".pt"):
            continue
        tau_val = fname.split("_tau_")[-1].replace(".pt", "")
        if tau_val == benchmark_tau:
            continue  # skip the benchmark itself

        opt_path = os.path.join(models_dir, env_name, fname)
        actor = load_policy(opt_path, opt_actor_class, state_dim, action_dim, device)
        V_model = evaluate_policy(actor, env, num_episodes)

        normalized_improvement = (V_model - V_random) / (V_opt_benchmark - V_random + 1e-8)

        results.append({
            "env": env_name,
            "data_size": 0,  # no data used for optimal models
            "tau": float(tau_val),
            "model": "optimal",
            "V_model": round(V_model, 2),
            "V_opt_tau_0.5": round(V_opt_benchmark, 2),
            "V_random": round(V_random, 2),
            "normalized_improvement": round(normalized_improvement, 4)
        })

    return results

env_n =   "LunarLander-v2"   # "CartPole-v0"   #

# ==== Run Evaluation ====
results = evaluate_all_models(
    trained_dir="trained_models",
    models_dir="models",
    learned_actor_class= learned_actor,
    opt_actor_class = opt_actor,
    env_name=env_n,
    num_episodes=30,
    device="cpu"
)


# Sort rows by env → data_size → tau → model
results.sort(key=itemgetter("env", "data_size", "tau", "model"))

# Define column order
column_order = ["env", "data_size", "tau", "model", "V_model", "V_opt_tau_0.5", "V_random", "normalized_improvement"]

# Reorder columns and print
ordered_results = [
    {col: row[col] for col in column_order}
    for row in results
]
pd.DataFrame(ordered_results).to_csv(f"final_results/{env_n}_evaluation_results.csv", index=False)
print(tabulate(ordered_results, headers="keys", tablefmt="github", floatfmt=".2f"))


