import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gym
from collections import namedtuple
import tqdm
import matplotlib.pyplot as plt
from buffer import ReplayBuffer
import pickle
import dill

# Define the neural network for Q-learning
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()

        # Define layers with ReLU activation
        self.linear1 = nn.Linear(input_size, 16)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(16, 16)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(16, 16)
        self.activation3 = nn.ReLU()

        # Output layer without activation function
        self.output_layer = nn.Linear(16, output_size)

        # Initialization using Xavier uniform
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        # Forward pass through the layers
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


class QNetwork:
    def __init__(self, env, lr):
        # Define Q-network with specified architecture
        self.net = FullyConnectedModel(4, 2)
        self.env = env
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def load_model(self, model_file):
        # Load pre-trained model from a file
        return self.net.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
        # Load pre-trained model weights from a file
        return self.net.load_state_dict(torch.load(weight_file))


class ReplayMemory:
    def __init__(self, env, memory_size=50000, burn_in=10000):
        # Initializes the replay memory, which stores transitions recorded from the agent taking actions in the environment.
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = []
        self.env = env

    def sample_batch(self, batch_size=32):
        # Returns a batch of randomly sampled transitions to be used for training the model.
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        # Appends a transition to the replay memory.
        self.memory.append(transition)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class EntropicUtility:
    def __init__(self, lamb=0.0):
        self.lamb = lamb

    def u(self, x):
        if self.lamb == 0:
            return x
        else:
            return (torch.exp(self.lamb * x) - 1) / self.lamb

    def dudx(self, x):
        if self.lamb == 0:
            return torch.ones_like(x)
        else:
            return torch.exp(self.lamb * x)


class DQN_Agent:

    def __init__(self, environment_name, lr=5e-4, lmda= 0.0):
        self.env = gym.make(environment_name)
        self.lr = lr
        self.policy_net = QNetwork(self.env, self.lr)
        self.target_net = QNetwork(self.env, self.lr)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())
        self.rm = ReplayMemory(self.env)
        self.burn_in_memory()
        self.batch_size = 32
        self.gamma = 0.99
        self.c = 0
        self.entropic_utility = EntropicUtility(lamb= lmda)  # Adjust lambda as needed

    def burn_in_memory(self):
        cnt = 0
        terminated = False
        truncated = False
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while cnt < self.rm.burn_in:
            if terminated or truncated:
                state = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            action = torch.tensor(random.sample([0, 1], 1)[0]).reshape(1, 1)
            next_state, reward, terminated, truncated = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def greedy_policy(self, q_values):
        return torch.argmax(q_values)

    def train(self):
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            q_values = self.policy_net.net(state)

            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)

            next_state, reward, terminated, truncated = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)

            state = next_state

            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = self.policy_net.net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(self.batch_size)

            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0]

            # Compute entropic utility values
            state_action_entropic_utility = self.entropic_utility.u(state_action_values)
            expected_entropic_utility = self.entropic_utility.u((next_state_values * self.gamma) + reward_batch)


            # Q-learning update with entropic risk
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_entropic_utility, expected_entropic_utility.unsqueeze(1))

            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

    def test(self):
        max_t = 1000
        state = self.env.reset()
        rewards = []

        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            state, reward, terminated, truncated = self.env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break

        return np.sum(rewards)

    def collect_data(self):
        state = self.env.reset()
        num_samples = 1000
        buffer = ReplayBuffer(buffer_size=100_000, batch_size=256, device = "cpu")
        for _ in range(num_samples):
            s = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net.net(s)
            action = self.greedy_policy(q_values).item()

            next_state, reward, done, _ = self.env.step(action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            if done:
                state = self.env.reset()

        return buffer




# Set environment and training parameters
env_name = 'CartPole-v0'
num_episodes_train = 200
num_episodes_test = 20
learning_rate = 5e-4

# Plot average performance of 5 trials
num_seeds = 1
l = num_episodes_train // 10
res = np.zeros((num_seeds, l))
gamma = 0.99

lambda_list = [-0.5,0,0.5]
agent_name = ['risk-averse', 'risk-neutral', 'risk-seeking']
for l in range(len(lambda_list)):
    agent = DQN_Agent(env_name, lr=learning_rate, lmda= lambda_list[l])
    # Training loop
    for m in range(num_episodes_train):
        agent.train()

        if m % 10 == 0:
            print("Episode: {}".format(m))
            # Evaluate the agent's performance over 20 test episodes
            G = np.zeros(num_episodes_test)
            for k in range(num_episodes_test):
                g = agent.test()
                G[k] = g
            reward_mean = G.mean()
            reward_sd = G.std()
            print(f"The test reward for episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")

    dataset = agent.collect_data()
    # Specify the file name where the object will be stored
    filename = str(env_name)+"-"+str(agent_name[l]+".pkl")  #str("~/PycharmProjects/CQL_AC/datasets/")+

    # Open the file in binary write mode and use pickle to dump the object
    with open(filename, 'wb') as file:
        dill.dump(dataset, file)

    print(f"Data has been pickled and saved to {filename}.")


import sys
sys.exit()


# Loop over multiple seeds
for i in tqdm.tqdm(range(num_seeds)):
    reward_means = []
    agent = DQN_Agent(env_name, lr=learning_rate, lmda= -0.5)

    # Training loop
    for m in range(num_episodes_train):
        agent.train()

        if m % 10 == 0:
            print("Episode: {}".format(m))

            # Evaluate the agent's performance over 20 test episodes
            G = np.zeros(num_episodes_test)
            for k in range(num_episodes_test):
                g = agent.test()
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            print(f"The test reward for episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")
            reward_means.append(reward_mean)

    res[i] = np.array(reward_means)


