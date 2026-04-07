import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from random import choices
from tqdm import tqdm

# Load the transition data from a CSV
df = pd.read_csv("/Users/abhilashchenreddy/PycharmProjects/CQL_AC/discrete_domains/domains/machine.csv")  # Replace with your file path

# Environment Class
class Environment:
    def __init__(self, df):
        self.transitions = defaultdict(list)
        self.rewards = defaultdict(dict)
        self.states = sorted(df["idstatefrom"].unique())
        self.actions = sorted(df["idaction"].unique())

        for _, row in df.iterrows():
            self.transitions[(row["idstatefrom"], row["idaction"])].append(
                (row["idstateto"], row["probability"])
            )
            self.rewards[(row["idstatefrom"], row["idaction"], row["idstateto"])] = row[
                "reward"
            ]

    def step(self, state, action):
        outcomes = self.transitions[(state, action)]
        next_state = choices([o[0] for o in outcomes], weights=[o[1] for o in outcomes])[0]
        reward = self.rewards[(state, action, next_state)]
        return next_state, reward

    def reset(self):
        return np.random.choice(self.states)

# Offline Data Buffer
class OfflineBuffer:
    def __init__(self):
        self.buffer = []

    def store(self, state, action, next_state, reward):
        self.buffer.append((state, action, next_state, reward))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# Q-learning Agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(len(env.actions)))
        self.alpha = alpha
        self.gamma = gamma

    def train(self, buffer, episodes):
        for _ in tqdm(range(episodes)):
            for state, action, next_state, reward in buffer.buffer:
                action_idx = action - 1
                max_next_q = max(self.q_table[next_state])
                td_target = reward + self.gamma * max_next_q
                self.q_table[state][action_idx] += self.alpha * (
                    td_target - self.q_table[state][action_idx]
                )

    def get_policy(self):
        policy = {}
        for state in self.env.states:
            policy[state] = np.argmax(self.q_table[state]) + 1
        return policy

# SAC-N Agent
class SACNAgent:
    def __init__(self, env, n_critics=3, gamma=0.9, beta=0.1, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.n_critics = n_critics

        self.q_tables = [defaultdict(lambda: np.zeros(len(env.actions))) for _ in range(n_critics)]
        self.policy = defaultdict(lambda: np.full(len(env.actions), 1 / len(env.actions)))

    def update_policy(self):
        for state in self.env.states:
            avg_q = np.mean([q_table[state] for q_table in self.q_tables], axis=0)
            logits = avg_q / self.beta
            exp_logits = np.exp(logits - np.max(logits))
            self.policy[state] = exp_logits / exp_logits.sum()

    def select_action(self, state):
        return np.random.choice(range(1, len(self.env.actions) + 1), p=self.policy[state])

    def update_q_tables(self, state, action, next_state, reward, done):
        action_idx = action - 1
        for q_table in self.q_tables:
            if done:
                target = reward
            else:
                min_next_q = np.min([qt[next_state] for qt in self.q_tables], axis=0)
                target = reward + self.gamma * np.dot(self.policy[next_state], min_next_q)
            q_table[state][action_idx] += self.alpha * (target - q_table[state][action_idx])

    def train(self, buffer, episodes):
        for _ in tqdm(range(episodes)):
            for state, action, next_state, reward in buffer.buffer:
                done = not self.env.transitions[(state, action)]
                self.update_q_tables(state, action, next_state, reward, done)
            self.update_policy()

    def get_policy(self):
        deterministic_policy = {}
        for state in self.env.states:
            avg_q = np.mean([q_table[state] for q_table in self.q_tables], axis=0)
            deterministic_policy[state] = np.argmax(avg_q) + 1
        return deterministic_policy

# Improved SAC-N Agent
class ImprovedSACNAgent(SACNAgent):
    def __init__(self, env, n_critics=3, gamma=0.9, beta=0.1, alpha=0.1):
        super().__init__(env, n_critics, gamma, beta, alpha)

    def update_q_tables(self, state, action, next_state, reward, done):
        action_idx = action - 1
        for q_table in self.q_tables:
            if done:
                target = reward
            else:
                min_next_q = np.min([qt[next_state] for qt in self.q_tables], axis=0)
                target = reward + self.gamma * np.dot(self.policy[next_state], min_next_q)
            q_table[state][action_idx] += self.alpha * (target - q_table[state][action_idx])

    def update_policy(self):
        for state in self.env.states:
            min_q = np.min([q_table[state] for q_table in self.q_tables], axis=0)
            logits = min_q / self.beta
            exp_logits = np.exp(logits - np.max(logits))
            self.policy[state] = exp_logits / exp_logits.sum()

    def get_policy(self):
        deterministic_policy = {}
        for state in self.env.states:
            min_q = np.min([q_table[state] for q_table in self.q_tables], axis=0)
            deterministic_policy[state] = np.argmax(min_q) + 1
        return deterministic_policy

# Evaluate a Policy
def evaluate_policy(env, policy, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(100):
            action = policy.get(state, np.random.choice(env.actions))
            next_state, reward = env.step(state, action)
            episode_reward += reward
            state = next_state

            if not env.transitions[(state, action)]:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

# Run Experiment with Evaluation
def run_experiment_with_evaluation(sample_sizes, episodes_per_size=100, num_eval_episodes=10, agent_type='q_learning'):
    results = {}
    avg_rewards = {}
    stochastic_policies = {}

    for sample_size in sample_sizes:
        print(f"Sample size: {sample_size}")
        env = Environment(df)
        buffer = OfflineBuffer()

        state = env.reset()
        for _ in range(sample_size):
            action = np.random.choice(env.actions)
            next_state, reward = env.step(state, action)
            buffer.store(state, action, next_state, reward)
            state = next_state

        if agent_type == 'q_learning':
            agent = QLearningAgent(env)
        elif agent_type == 'sacn':
            agent = SACNAgent(env)
        elif agent_type == 'improved_sacn':
            agent = ImprovedSACNAgent(env)
        else:
            raise ValueError("Invalid agent type. Choose 'q_learning', 'sacn' or 'improved_sacn'.")

        agent.train(buffer, episodes_per_size)

        if agent_type in ['sacn', 'improved_sacn']:
            stochastic_policies[agent_type] = {
                state: agent.policy[state] for state in env.states
            }

        policy = agent.get_policy()
        results[sample_size] = policy

        avg_reward = evaluate_policy(env, policy, num_eval_episodes)
        avg_rewards[sample_size] = avg_reward

    if stochastic_policies:
        plot_stochastic_policies(stochastic_policies, env)

    return avg_rewards


def plot_stochastic_policies(stochastic_policies, env):
    actions = env.actions
    states = env.states

    for action_idx, action in enumerate(actions):
        plt.figure(figsize=(10, 6))
        for model_name, policy in stochastic_policies.items():
            probabilities = [policy[state][action_idx] for state in states]
            plt.plot(states, probabilities, label=f"{model_name}")

        plt.title(f"Action {action} Probabilities Across States")
        plt.xlabel("State ID")
        plt.ylabel("Action Probability")
        plt.legend()
        plt.grid()
        plt.show()


# Example of Running the Experiment
sample_sizes = [10]  #, 500, 1000]
avg_rewards = run_experiment_with_evaluation(sample_sizes, agent_type='sacn')
print(avg_rewards)
avg_rewards = run_experiment_with_evaluation(sample_sizes, agent_type='improved_sacn')
print(avg_rewards)
