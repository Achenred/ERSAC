# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import gym
# import numpy as np
# import random
# from collections import deque
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Q-network
# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.out = nn.Linear(128, action_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.out(x)
#
# # Replay buffer
# class ReplayBuffer:
#     def __init__(self, capacity=50000):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, transition):
#         self.buffer.append(transition)
#
#     def sample(self, batch_size):
#         return random.sample(self.buffer, batch_size)
#
#     def __len__(self):
#         return len(self.buffer)
#
# # Expectile computation
# def compute_expectile(values, tau, tol=1e-4, max_iter=500):
#     low, high = values.min().item(), values.max().item()
#     for _ in range(max_iter):
#         mid = (low + high) / 2
#         diff = values - mid
#         score = torch.mean(2 * ((1 - tau) * torch.clamp(diff, min=0) - tau * torch.clamp(-diff, min=0)))
#         if abs(score.item()) < tol:
#             return torch.tensor(mid, device=values.device)
#         if score.item() > 0:
#             low = mid
#         else:
#             high = mid
#     return torch.tensor(mid, device=values.device)
#
# # Simulate next state using environment copy
# def simulate_td_targets(env_sim, state, action, q_target, gamma, n_samples=5):
#     td_targets = []
#     for _ in range(n_samples):
#         env_sim.reset()
#         env_sim.state = np.copy(state)
#         next_state, reward, done, _ = env_sim.step(action)
#         next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
#         with torch.no_grad():
#             max_q = q_target(next_tensor).max().item()
#         td = reward + gamma * (0 if done else max_q)
#         td_targets.append(td)
#     return torch.tensor(td_targets, dtype=torch.float32, device=device)
#
# # Training loop
# def train_dqn(
#     env_name='CartPole-v0',
#     use_expectile=True,
#     tau=0.9,
#     n_episodes=500,
#     gamma=0.99,
#     lr=1e-3,
#     batch_size=64,
#     n_samples=10,
#     update_target_every=10, update_param = 0.01
# ):
#     env = gym.make(env_name)
#     env_sim = gym.make(env_name)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#
#     q_net = QNetwork(state_dim, action_dim).to(device)
#     q_target = QNetwork(state_dim, action_dim).to(device)
#     q_target.load_state_dict(q_net.state_dict())
#     optimizer = optim.Adam(q_net.parameters(), lr=lr)
#     buffer = ReplayBuffer()
#
#     epsilon_start = 0.2
#     epsilon_final = 0.05
#     epsilon_decay = 0.05
#     epsilon = epsilon_start
#
#     for ep in range(n_episodes):
#         if ep % 50 == 0:
#             epsilon = max(epsilon_final, epsilon - epsilon_decay)
#         state = env.reset()
#         total_reward = 0
#         done = False
#
#         while not done:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#             if random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 with torch.no_grad():
#                     action = q_net(state_tensor).argmax().item()
#
#             next_state, reward, done, _ = env.step(action)
#             buffer.push((state, action, reward, next_state, done))
#             state = next_state
#             total_reward += reward
#
#             if len(buffer) < batch_size:
#                 continue
#
#             transitions = buffer.sample(batch_size)
#             batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*transitions)
#             batch_states = torch.FloatTensor(batch_states).to(device)
#             batch_actions = torch.LongTensor(batch_actions).to(device)
#             batch_next_states = torch.FloatTensor(batch_next_states).to(device)
#             batch_dones = torch.FloatTensor(batch_dones).to(device)
#             batch_rewards = torch.FloatTensor(batch_rewards).to(device)
#
#             targets = []
#             for i in range(batch_size):
#                 if use_expectile:
#                     s = batch_states[i].cpu().numpy()
#                     a = batch_actions[i].item()
#                     td_samples = simulate_td_targets(env_sim, s, a, q_target, gamma, n_samples)
#                     target = compute_expectile(td_samples, tau)
#                 else:
#                     with torch.no_grad():
#                         max_q = q_target(batch_next_states[i].unsqueeze(0)).max().item()
#                     target = batch_rewards[i] + gamma * (1 - batch_dones[i]) * max_q
#                 targets.append(target)
#
#             targets = torch.stack(targets).detach()
#             q_values = q_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze()
#             loss = F.mse_loss(q_values, targets)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if ep % update_target_every == 0:
#             for target_param, source_param in zip(q_target.parameters(), q_net.parameters()):
#                 target_param.data.copy_(update_param * source_param.data + (1.0 - update_param) * target_param.data)
#
#         print(f"[{env_name}] Ep {ep:03d} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
#
#     env.close()
#     env_sim.close()
#
#
# # === Run Script ===
# if __name__ == "__main__":
#     # Set to True for Expectile-DQN, False for Standard DQN
#     train_dqn(
#         env_name= 'LunarLander-v2',    #'CartPole-v0',
#         use_expectile=False,      # Toggle here!
#         tau=0.5,                  # Only relevant if use_expectile=True
#         n_episodes=300,
#         gamma=0.99,
#         lr=1e-4,
#         batch_size=64,
#         n_samples=5,
#         update_target_every=10
#     )


import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.logits(x), dim=-1)

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x)

# Expectile risk computation via bisection
def compute_expectile(values, tau, tol=1e-4, max_iter=500):
    low, high = values.min().item(), values.max().item()
    for _ in range(max_iter):
        mid = (low + high) / 2
        diff = values - mid
        score = torch.mean(2 * ((1 - tau) * torch.clamp(diff, min=0) - tau * torch.clamp(-diff, min=0)))
        if abs(score.item()) < tol:
            return torch.tensor(mid, device=values.device)
        if score.item() > 0:
            low = mid
        else:
            high = mid
    return torch.tensor(mid, device=values.device)

# Training loop
def train_actor_critic(
    env_name = "LunarLander-v2",
    n_episodes=1000,
    gamma=0.99,
    lr=1e-4,
    use_expectile=False,
    tau=0.7,
    n_bootstrap=10
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)
    optimizerA = optim.Adam(actor.parameters(), lr=lr)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)

    for ep in range(n_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        done = False
        total_reward = 0

        while not done:
            # Actor selects action
            probs = actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step in env
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

            # Critic target
            if use_expectile:
                # bootstrap n times with noisy transitions
                td_samples = []
                for _ in range(n_bootstrap):
                    with torch.no_grad():
                        v_next = critic(next_state_tensor).item()
                        td = reward + gamma * v_next * (1 - int(done))
                        td_samples.append(td)
                td_tensor = torch.tensor(td_samples, dtype=torch.float32, device=device)
                target = compute_expectile(td_tensor, tau)
            else:
                with torch.no_grad():
                    v_next = critic(next_state_tensor)
                    target = reward_tensor + gamma * (1 - int(done)) * v_next

            # Critic loss and update
            value = critic(state)
            critic_loss = (value - target).pow(2).mean()

            optimizerC.zero_grad()
            critic_loss.backward()
            optimizerC.step()

            # Actor loss and update
            advantage = (target - value).detach()
            actor_loss = -log_prob * advantage

            optimizerA.zero_grad()
            actor_loss.backward()
            optimizerA.step()

            # Transition
            state = next_state_tensor

        print(f"[Ep {ep}] Reward: {total_reward:.2f}")

    env.close()

# === Run ===
if __name__ == "__main__":
    train_actor_critic(
        env_name= "",   #"LunarLander-v2",
        n_episodes=1000,
        gamma=0.99,
        lr=1e-4,
        use_expectile=False,  # <== Toggle this flag
        tau=0.7,
        n_bootstrap=10
    )
