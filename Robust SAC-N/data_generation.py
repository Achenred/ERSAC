import os
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.logits(x), dim=-1)

# Critic network (estimates Q(s, a))
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, 1)
        self.action_dim = action_dim

    def forward(self, state, action):
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([state, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_out(x)


def expectile_loss(y_true, y_pred, tau):
    diff = y_true - y_pred
    weight = torch.where(diff < 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()

def compute_returns(next_value, rewards, masks, gamma=0.9):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

# def train(actor, critic, tau, env, n_iters=10000, lr=1e-3, gamma=0.9):
#     optimizerA = optim.Adam(actor.parameters(), lr=lr)
#     optimizerC = optim.Adam(critic.parameters(), lr=lr*10)
#
#     for iter in tqdm(range(n_iters), desc=f"Training for tau={tau}"):
#         state = env.reset()
#         log_probs, values, rewards, masks = [], [], [], []
#
#         for t in count():
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#             dist = actor(state_tensor)
#             value = critic(state_tensor)
#
#             action = dist.sample()
#             next_state, reward, done, _ = env.step(action.item())
#
#             log_probs.append(dist.log_prob(action))
#             values.append(value)
#             rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
#             masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))
#
#             state = next_state
#             if done:
#                 break
#
#         next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
#         next_value = critic(next_state_tensor)
#         returns = compute_returns(next_value, rewards, masks, gamma)
#
#         log_probs = torch.stack(log_probs)
#         values = torch.cat(values)
#         returns = torch.cat(returns).detach()
#
#         advantage = returns - values
#         actor_loss = -(log_probs * advantage.detach()).mean()
#         critic_loss = expectile_loss(returns, values, tau)
#
#         optimizerA.zero_grad()
#         optimizerC.zero_grad()
#         actor_loss.backward()
#         critic_loss.backward()
#         optimizerA.step()
#         optimizerC.step()
#
#         if iter % 50 == 0:
#             total_reward = sum(r.item() for r in rewards)
#             print(
#                 f"[Iter {iter:04d}] Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | Episode Reward: {total_reward:.2f}")

def train(actor, critic, tau, env, n_iters=3000, lr=0.0005, gamma=0.9):
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)

    for iter in range(n_iters):
        state = env.reset()
        log_probs, values, rewards, masks = [], [], [], []

        for t in count():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist = actor(state_tensor)
            value = critic(state_tensor)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

            state = next_state
            if done:
                episode_reward = sum(r.item() for r in rewards)
                print(f"Iteration {iter}, Steps: {t}, Total Reward: {episode_reward:.2f}")
                break

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        next_value = critic(next_state_tensor)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.stack(log_probs)
        values = torch.cat(values)
        returns = torch.cat(returns).detach()

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = expectile_loss(returns, values, tau)

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()

def evaluate_policy(actor, env, num_episodes=20, max_steps=500):
    actor.eval()
    total_reward = 0.0
    with torch.no_grad():
        for _ in tqdm(range(num_episodes), desc="Evaluating Policy"):
            state = env.reset()
            episode_reward = 0.0
            for _ in range(max_steps):
                state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
                dist = actor(state_tensor)
                action = dist.probs.argmax(dim=-1).item()
                next_state, reward, done, _ = env.step(action)
                # print(action, reward)
                state = next_state
                episode_reward += reward
                if done:
                    break
            print(episode_reward)
            total_reward += episode_reward
    avg_reward = total_reward / num_episodes
    print(f"Evaluation: Average reward over {num_episodes} episodes = {avg_reward:.2f}")
    return avg_reward




# def collect_samples(actor, env, num_samples):
#     data = []
#     state = env.reset()
#     for _ in tqdm(range(num_samples), desc="Collecting Samples"):
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#         dist = actor(state_tensor)
#         print(dist)
#         import sys
#         sys.exit()
#         action = dist.sample().item()
#         next_state, reward, done, _ = env.step(action)
#         data.append((state, action, reward, next_state, done))
#         state = next_state if not done else env.reset()
#     return data

import torch
from torch.distributions import Categorical
from tqdm import tqdm

def collect_samples(actor, env, num_samples):
    """Roll the policy for `num_samples` steps and return a list of transitions."""
    actor.eval()                          # inference mode (no dropout, etc.)
    data   = []
    state  = env.reset()

    for _ in tqdm(range(num_samples), desc="Collecting Samples"):

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = actor(state_tensor)                  # shape [1, A]

        dist   = Categorical(probs=action_probs)
        action = dist.sample().item()

        next_state, reward, done, _ = env.step(action)

        data.append((state.copy(), action, reward, next_state.copy(), done))

        state = next_state if not done else env.reset()

    actor.train()                         # switch back to training mode
    return data


# === Main Script ===
env_name =  "LunarLander-v2" #"CartPole-v0"
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

risk_profiles = [0.1, 0.5, 0.9]
sample_sizes = [1000, 10000, 100000]

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

for tau in risk_profiles:
    actor_path = f"models/{env_name}/best_actor_tau_{tau}.pt"
    critic_path = f'models/{env_name}/best_critic_tau_{tau}.pt'

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size,action_size).to(device)

    if os.path.exists(actor_path) and os.path.exists(critic_path):
        print(f"Loading pretrained models for tau={tau}")
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))
    else:
        break
        # print(f"Training models for tau={tau}")
        # train(actor, critic, tau=tau, env=env, n_iters=3000)
        # torch.save(actor.state_dict(), actor_path)
        # torch.save(critic.state_dict(), critic_path)


    # evaluate_policy(actor, env, num_episodes=50, max_steps=1000)

    for sample_size in sample_sizes:
        data_path = f"data/{env_name}_samples_tau_{tau}_n_{sample_size}.pkl"
        if os.path.exists(data_path):
            print(f"Sample data already exists: {data_path}")
            continue
        samples = collect_samples(actor, env, sample_size)
        df = pd.DataFrame(samples, columns=["state", "action", "reward", "next_state", "done"])
        df.to_pickle(data_path)
