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
#         env_name= 'CartPole-v0',   #'LunarLander-v2',    #'CartPole-v0',
#         use_expectile=False,      # Toggle here!
#         tau=0.5,                  # Only relevant if use_expectile=True
#         n_episodes=1000,
#         gamma=0.99,
#         lr=1e-3,
#         batch_size=64,
#         n_samples=5,
#         update_target_every=10
#     )

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor network (outputs policy)
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

# Expectile computation (bisection)
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

def sample_td_targets(env_name, state, action, critic, gamma, n_samples=10):
    td_targets = []
    for _ in range(n_samples):
        env_copy = gym.make(env_name)
        env_copy.reset()
        env_copy.state = np.copy(state)
        next_state, reward, done, _ = env_copy.step(action)

        with torch.no_grad():
            next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)


            q_next = torch.stack([
                critic(next_tensor, torch.tensor([a], device=device))
                for a in range(env_copy.action_space.n)
            ])
            target = reward + gamma * (0 if done else q_next.max().item())
        td_targets.append(target)

        env_copy.close()

    return torch.tensor(td_targets, dtype=torch.float32, device=device)


def train_actor_critic_q(
    env_name='CartPole-v0',
    n_episodes=500,
    gamma=0.99,
    lr=3e-4,
    use_expectile=True,
    tau=0.7,
    n_bootstrap=10,
    update_factor = 0.01, load_pretrained = False
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n


    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    actor_ckpt = f"models/{env_name}/best_actor_tau_0.pt"
    critic_ckpt = f"models/{env_name}/best_critic_tau_0.pt"
    if load_pretrained and os.path.isfile(actor_ckpt) and os.path.isfile(critic_ckpt):
        actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
        critic.load_state_dict(torch.load(critic_ckpt, map_location=device))
        print(f"loaded pretrained weights from '{actor_ckpt}', '{critic_ckpt}'")

    target_critic = Critic(state_dim, action_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())  # hard copy


    optimizerA = optim.Adam(actor.parameters(), lr=lr*0.5)
    optimizerC = optim.Adam(critic.parameters(), lr=lr)

    K = 10  # window size for the running average
    recent_returns = collections.deque(maxlen=K)
    best_avg_return = -float("inf")  # sentinel

    BEST_LOG = "best_models.txt"
    with open(BEST_LOG, "w") as f:
        f.write("env, tau, episode, avg_return\n")

    for ep in range(n_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        done = False
        total_reward = 0

        critic_loss_sum = 0  #torch.tensor(0.0, device=device)
        actor_loss_sum = 0   #torch.tensor(0.0, device=device)
        step_count = 0

        optimizerC.zero_grad()
        optimizerA.zero_grad()

        while not done:
            # Sample action from policy
            probs = actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step in environment
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)

            if use_expectile:

                td_tensor = sample_td_targets(
                    env_name,
                    state.squeeze(0).cpu().numpy(),
                    action.item(),
                    target_critic,
                    gamma,
                    n_samples=n_bootstrap
                )

                td_target = compute_expectile(td_tensor, tau)
            else:
                with torch.no_grad():
                    q_next = torch.stack([
                        target_critic(next_state_tensor, torch.tensor([a], device=device))
                        for a in range(action_dim)
                    ])
                    td_target = reward_tensor + gamma * (0 if done else q_next.max().item())

            # Critic update
            q_pred = critic(state, action).view(1)


            critic_loss = F.mse_loss(q_pred, td_target.view(1))
            critic_loss_sum += critic_loss


            # optimizerC.zero_grad()
            # critic_loss.backward()
            # optimizerC.step()

            # Actor update: logπ(a|s) * Q(s,a)
            q_val = critic(state, action).detach()
            with torch.no_grad():
                q_next = torch.stack([
                    critic(next_state_tensor, torch.tensor([a], device=device))
                    for a in range(action_dim)
                ]).squeeze()  # [A]
                v_s = q_next.max()
            adv = q_val - v_s

            # actor_loss = -(log_prob * adv)

            actor_loss = -log_prob * q_val

            actor_loss_sum += actor_loss
            step_count += 1



            # optimizerA.zero_grad()
            # actor_loss.backward()
            # optimizerA.step()

            state = next_state_tensor

        critic_loss_mean = critic_loss_sum.squeeze() / step_count
        actor_loss_mean = actor_loss_sum.squeeze() / step_count

        critic_loss_mean.backward()
        actor_loss_mean.backward()

        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)  # optional
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)

        optimizerC.step()
        optimizerA.step()

        with torch.no_grad():
            for p_src, p_tgt in zip(critic.parameters(), target_critic.parameters()):
                p_tgt.mul_(1.0 - update_factor).add_(update_factor * p_src)

        recent_returns.append(total_reward)
        if len(recent_returns) == K:
            avg_return = sum(recent_returns) / K
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                torch.save(actor.state_dict(), f"models/{env_name}/best_actor_tau_{tau}.pt")
                torch.save(critic.state_dict(), f"models/{env_name}/best_critic_tau_{tau}.pt")
                print(f"Ep {ep} | new best {best_avg_return:.1f}  —  models saved")

                with open(BEST_LOG, "a") as f:
                    f.write(f"{env_name}, {tau}, {ep}, {best_avg_return:.2f}\n")

        print(f"[{env_name}] Episode {ep} | Total Reward: {total_reward:.2f}")

    env.close()

# === Run ===
if __name__ == "__main__":
    envs = ["CartPole-v0"]   #["CartPole-v0", "LunarLander-v2"]
    taus = [0.9] # [0.5, 0.1, 0.9]

    lr_map = {
        "CartPole-v0": 1e-4,  # lr for CartPole
        "LunarLander-v2": 2e-4  # lr for LunarLander
    }

    for env_name in envs:
        for tau in taus:
            print(f"\n=== {env_name} | τ = {tau} | lr = {lr_map[env_name]} ===")
            train_actor_critic_q(
                env_name=env_name,
                n_episodes=10000,
                gamma=0.99,
                lr=lr_map[env_name],
                use_expectile=True,
                tau=tau,
                n_bootstrap=100,
                update_factor= 0.01,
                load_pretrained= True
            )
