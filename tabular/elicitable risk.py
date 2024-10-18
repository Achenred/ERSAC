import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import gym


# Function to calculate expectile loss
def expectile_loss(q, X, tau):
    positive_part = np.maximum(0, q - X) ** 2
    negative_part = np.maximum(0, X - q) ** 2
    loss = (1 - tau) * np.mean(positive_part) + tau * np.mean(negative_part)
    return loss


# Function to calculate expectile given X and tau
def calculate_expectile(X, tau):
    result = minimize(expectile_loss, x0=np.mean(X), args=(X, tau))
    return result.x[0]


# Function to generate Dirichlet distributed transition matrices
def generate_transition_matrices(num_matrices, num_states, num_actions):
    matrices = []
    for _ in range(num_matrices):
        matrix = np.random.dirichlet(np.ones(num_states * num_actions), size=num_states)
        matrix = matrix.reshape(num_states, num_actions, num_states)
        matrices.append(matrix)
    return matrices


# Function to sample transitions from a transition matrix
def sample_transitions(env, transition_matrix, num_samples=100):
    transitions = []
    for _ in range(num_samples):
        s = env.reset()  # Reset environment and get initial state
        a = np.random.choice(np.arange(env.action_space.n), p=transition_matrix[s].sum(axis=1))  # Sample action
        s_next, r, done, _ = env.step(a)  # Take a step in the environment
        transitions.append((s, a, r, s_next))  # Record the transition
        if done:
            break  # Reset if episode ends
    return transitions


# Function to estimate Q-table using dynamic expectile risk with sampled transitions
def estimate_q_table_with_expectile_risk(env, num_matrices, num_states, num_actions, num_episodes=1000, alpha=0.1,
                                         gamma=0.9, tau=0.5):
    # Generate transition matrices
    transition_matrices = generate_transition_matrices(num_matrices, num_states, num_actions)

    # Initialize Q-table with zeros
    # Initialize Q-tables for each transition matrix
    Q_tables = [np.zeros((num_states, num_actions)) for _ in range(num_matrices)]

    # Q-learning with dynamic expectile risk and sampled transitions
    for episode in tqdm(range(num_episodes), desc='Q-learning with Dynamic Expectile Risk'):
        for idx, transition_matrix in enumerate(transition_matrices):
            # Sample transitions from selected transition matrix
            transitions = sample_transitions(env, transition_matrix)
            for (s, a, r, s_next) in transitions:
                # Calculate next state values for dynamic expectile risk
                next_state_values = Q_tables[idx][s_next, :]
                expectile_value = calculate_expectile(next_state_values, tau)

                # Update Q-table using Q-learning update rule with expectile risk
                Q_tables[idx][s, a] = (1 - alpha) * Q_tables[idx][s, a] + alpha * (r + gamma * expectile_value)

    return Q_tables


# Example usage
# Assuming we have an OpenAI Gym environment 'env' with discrete states and actions
env = gym.make('Taxi-v3')
num_matrices = 3  # Number of transition matrices to generate
num_states = env.observation_space.n  # Number of states in the environment
num_actions = env.action_space.n  # Number of actions in the environment
num_episodes = 10  # Number of episodes for Q-learning
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
tau = 0.7  # Risk parameter for dynamic expectile risk

# Estimate Q-table using dynamic expectile risk with sampled transitions
Q_table = estimate_q_table_with_expectile_risk(env, num_matrices, num_states, num_actions, num_episodes, alpha, gamma,
                                               tau)
# Print or visualize Q-table
print("Estimated Q-table:")


# Offline data replay buffer D (assuming pre-filled)
D = []
# Function to generate transitions and add to replay buffer D
def generate_transitions(env, num_samples=10000):
    transitions = []
    for _ in range(num_samples):
        s = env.reset()
        while True:
            a = env.action_space.sample()
            s_next, r, done, _ = env.step(a)
            transitions.append((s, a, r, s_next))
            if done:
                break
            s = s_next
    return transitions

# Populate replay buffer D with transitions
D = generate_transitions(env)
batch_size = 32
# Sample a mini-batch B from the replay buffer D
mini_batch = np.random.choice(D, batch_size)
# Function to compute target Q-values
def compute_target_q_values(mini_batch, Q_tables, beta, gamma, tau):
    target_q_values = []
    for (s, a, r, s_next) in mini_batch:
        # Calculate target Q-values for each Q-table
        next_state_values = np.array([Q_table[s_next, :] for Q_table in Q_tables])
        # Calculate expectile value for each action
        expectile_values = [calculate_expectile(next_state_values[:, action], tau) for action in range(num_actions)]
        # Compute min Q-value using expectile
        min_expectile_value = np.min(expectile_values)
        # Assume a uniform policy for simplicity: pi_theta(a'|s') = 1/num_actions
        log_pi_theta = np.log(1 / num_actions)

        # Compute the target Q-value
        target_q_value = r + gamma * min_expectile_value - beta * log_pi_theta

        target_q_values.append(target_q_value)
    return target_q_values


# Compute target Q-values
target_q_values = compute_target_q_values(mini_batch, target_Q_tables, beta, gamma, tau)

# Print computed target Q-values for inspection
print(target_q_values)
import sys
sys.exit()












# Q-learning with dynamic expectile risk and EDAC
for episode in tqdm(range(num_episodes), desc='Q-learning with Dynamic Expectile Risk'):
    # Sample a mini-batch B from the replay buffer D
    batch_size = 64
    mini_batch = np.random.choice(D, batch_size)

    for (s, a, r, s_next) in mini_batch:
        for i in range(num_matrices):
            # Compute target Q-value using dynamic expectile risk
            next_state_values = target_Q_tables[i][s_next, :]
            expectile_value = calculate_expectile(next_state_values, tau)
            target = r + gamma * expectile_value

            # Update Q-table using Q-learning update rule
            Q_tables[i][s, a] = (1 - alpha) * Q_tables[i][s, a] + alpha * target

    # Update target Q-tables
    for i in range(N):
        target_Q_tables[i] = Q_tables[i].copy()
