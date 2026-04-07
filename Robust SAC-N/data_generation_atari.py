# import gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from gym.wrappers import AtariPreprocessing, FrameStack
# from itertools import count
# import numpy as np
# import os
#
# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Create Atari environment with frame skipping disabled
# env = gym.make("SpaceInvaders-v0", frameskip=1, render_mode="rgb_array")
# env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)  # outputs (84, 84)
# env = FrameStack(env, num_stack=4)  # outputs (4, 84, 84)
#
# # Hyperparameters
# action_size = env.action_space.n
# lr = 0.0001
# gamma = 0.99
#
# # CNN Actor network
# class CNNActor(nn.Module):
#     def __init__(self, action_size):
#         super(CNNActor, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(3136, 512), nn.ReLU(),
#             nn.Linear(512, action_size)
#         )
#
#     def forward(self, x):
#         x = x / 255.0  # normalize pixel values
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         logits = self.fc(x)
#         return Categorical(F.softmax(logits, dim=-1))
#
# # CNN Critic network
# class CNNCritic(nn.Module):
#     def __init__(self):
#         super(CNNCritic, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(3136, 512), nn.ReLU(),
#             nn.Linear(512, 1)
#         )
#
#     def forward(self, x):
#         x = x / 255.0
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)
#
# # Return calculator
# def compute_returns(next_value, rewards, masks, gamma=0.99):
#     R = next_value
#     returns = []
#     for step in reversed(range(len(rewards))):
#         R = rewards[step] + gamma * R * masks[step]
#         returns.insert(0, R)
#     return returns
#
# # Training loop
# def train(actor, critic, n_iters=100):
#     optimizerA = optim.Adam(actor.parameters(), lr=lr)
#     optimizerC = optim.Adam(critic.parameters(), lr=lr)
#
#     for iter in range(n_iters):
#         state = env.reset()[0]
#         log_probs, values, rewards, masks = [], [], [], []
#         entropy = 0
#
#         for step in count():
#             state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
#             dist = actor(state_tensor)
#             value = critic(state_tensor)
#
#             action = dist.sample()
#             next_state, reward, terminated, truncated, _ = env.step(action.item())
#             done = terminated or truncated
#
#             log_prob = dist.log_prob(action).unsqueeze(0)
#             entropy += dist.entropy().mean()
#
#             log_probs.append(log_prob)
#             values.append(value)
#             rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
#             masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))
#
#             state = next_state
#
#             if done:
#                 print(f"Iteration {iter} | Steps: {step}")
#                 break
#
#         next_state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
#         next_value = critic(next_state_tensor)
#         returns = compute_returns(next_value, rewards, masks)
#
#         log_probs = torch.cat(log_probs)
#         returns = torch.cat(returns).detach()
#         values = torch.cat(values)
#
#         advantage = returns - values
#
#         actor_loss = -(log_probs * advantage.detach()).mean()
#         critic_loss = advantage.pow(2).mean()
#
#         optimizerA.zero_grad()
#         optimizerC.zero_grad()
#         actor_loss.backward()
#         critic_loss.backward()
#         optimizerA.step()
#         optimizerC.step()
#
#     torch.save(actor.state_dict(), "actor_spaceinvaders.pt")
#     torch.save(critic.state_dict(), "critic_spaceinvaders.pt")
#     env.close()
#
# # Main execution
# if __name__ == "__main__":
#     actor = CNNActor(action_size).to(device)
#     critic = CNNCritic().to(device)
#     train(actor, critic, n_iters=100)


import gym
import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return Categorical(F.softmax(logits, dim=-1))

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

def expectile_loss(y_true, y_pred, tau):
    diff = y_true - y_pred
    weight = torch.where(diff < 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(actor, critic,tau, n_iters):
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)

    for iter in range(n_iters):
        state = env.reset()
        log_probs, values, rewards, masks = [], [], [], []

        for t in count():
            # print(iter, t)
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
        # print(len(returns))
        # print(returns)
        # import sys
        # sys.exit()

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

if __name__ == "__main__":
    env_name =  "LunarLander-v2"  #"CartPole-v0"  #"LunarLander-v2"
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    lr = 0.0005
    gamma = 0.99
    tau = 0.9  # Set tau < 0.5 for risk-seeking, > 0.5 for risk-averse

    # Save/load paths
    actor_path = f"models/{env_name}/actor_tau_{tau}.pt"
    critic_path = f'models/{env_name}/critic_tau_{tau}.pt'

    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    os.makedirs(os.path.dirname(critic_path), exist_ok=True)

    # Initialize or load
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)
    # if os.path.exists(actor_path): actor.load_state_dict(torch.load(actor_path))
    # if os.path.exists(critic_path): critic.load_state_dict(torch.load(critic_path))

    train(actor, critic,tau, n_iters=5000)   #2500)

    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    env.close()
