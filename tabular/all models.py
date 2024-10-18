import numpy as np
import matplotlib.pyplot as plt

def expectileRisk(tau):
    def u_f(y):
        return 2*((1 - tau) * np.maximum(0, y) - tau * np.maximum(0, -y))

    def rho_int(p, v, u_f):
        tol = np.sqrt(np.finfo(float).eps)

        bounds = [np.min(v), np.max(v)]

        while bounds[1] - bounds[0] > tol:
            mid = np.mean(bounds)
            tmp = np.dot(p, u_f((v - mid)))
            if tmp >= 0:
                bounds[0] = mid
            else:
                bounds[1] = mid

        return np.mean(bounds)

    def rho_f(p, v):
        return rho_int(p, v, u_f)

    return rho_f, u_f

def rand_multi(v, n):
    inds = np.argsort(v)
    tmp = v[inds]
    xs = rand_multi_int(tmp, n)
    xs = inds[xs]
    return xs

def rand_multi_int(v, n):
    v = np.maximum(0, v)
    v = v / np.sum(v)
    tmp = np.random.rand(n)
    xs = np.ones(n, dtype=int)
    for k in range(len(v) - 1):
        tmp = tmp - v[k]
        xs[tmp > 0] = k + 2
    return xs - 1  # subtracting 1 to match 0-based index in Python


S = 5
A = 2
np.random.seed(0)
P = np.random.rand(S, A, S)
P /= P.sum(axis=2, keepdims=True)

r = np.random.rand(S, A)

gamma = 0.99
r *= (1 - gamma)

risk_tau = 0.75
rho_f, u_f = expectileRisk(risk_tau)

itN = 1000

# Q-iteration
Q = np.zeros((S, A)) #np.random.rand(S, A)
Q_ = np.zeros((S, A))
hs = []

for it in range(itN):
    for s in range(S):
        for a in range(A):
            P_sa = P[s, a, :]
            Q_sa = r[s, a] + gamma * np.max(Q, axis=1)
            Q_[s, a] = rho_f(P_sa, Q_sa)


    hs.append(np.linalg.norm(Q - Q_))
    Q = Q_.copy()

Q1 = Q.copy()
plt.figure()
plt.plot(hs)
plt.show()



# Function to derive policy from Q-values
def get_policy(Q):
    policy = np.zeros((S, A))
    for s in range(S):
        policy[s, np.argmax(Q[s])] = 1.0
    return policy

# Evaluation function
def evaluate_policy(policy, num_episodes=100):
    returns = []
    for _ in range(num_episodes):
        state = np.random.randint(S)
        total_return = 0
        done = False
        while not done:
            action = np.random.choice(A, p=policy[state])
            next_state = np.random.choice(S, p=P[state, action])
            reward = r[state, action]
            total_return += reward
            if np.random.rand() < 0.1:  # Assuming a terminal state with a 10% chance
                done = True
            state = next_state
        returns.append(total_return)
    return returns

# Derive the policy from Q-values
policy = get_policy(Q)

# Evaluate the policy
num_episodes = 100
returns = evaluate_policy(policy, num_episodes)

# Print the results
print(f"Average return over {num_episodes} episodes: {np.mean(returns)}")
print(f"Standard deviation of return: {np.std(returns)}")

import numpy as np
import random


# Assume you have the following variables already defined
# Q1: Final Q-table from the Q-iteration algorithm
# S: Number of states
# A: Number of actions
# P: Transition probability matrix, shape (S, A, S)
# r: Reward matrix, shape (S, A)
# gamma: Discount factor

# Function to choose action using epsilon-greedy policy
def choose_action(state, Q, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return np.random.randint(A)  # Explore: Random action
    else:
        return np.argmax(Q[state])  # Exploit: Action with highest Q-value


# Function to simulate a step in the environment
def simulate_step(state, action, P, r):
    next_state = np.random.choice(S, p=P[state, action])
    reward = r[state, action]
    done = False  # Assuming no terminal states in this example
    return next_state, reward, done


# Generate buffer data
def generate_buffer(Q, P, r, num_episodes=100, max_steps_per_episode=100, epsilon=0.1):
    buffer = []

    for episode in range(num_episodes):
        state = np.random.randint(S)  # Start from a random state
        for step in range(max_steps_per_episode):
            action = choose_action(state, Q, epsilon)
            next_state, reward, done = simulate_step(state, action, P, r)
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break

    return buffer


# Generate the buffer
buffer = generate_buffer(Q1, P, r)

# Example of buffer data structure: [(state, action, reward, next_state, done), ...]
print(buffer[:10000])  # Print the first 10 transitions for verification

# import numpy as np

# # Example buffer data generated previously
# buffer = generate_buffer(Q1, P, r, num_episodes=10000)

# Parameters
S = 5  # Number of states
A = 2  # Number of actions
gamma = 0.99  # Discount factor
alpha = 0.1  # Learning rate
cql_alpha = 1.0  # Regularization strength for CQL

# Initialize Q-table
Q = np.zeros((S, A))


# Helper function to compute CQL regularization term
def compute_cql_regularization(Q, state, cql_alpha):
    logsumexp_Q = np.log(np.sum(np.exp(Q[state])))  # logsumexp over all actions
    return cql_alpha * logsumexp_Q


# Offline CQL
for state, action, reward, next_state, done in buffer:
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state, best_next_action] * (1 - done)
    td_error = td_target - Q[state, action]

    # Add CQL regularization term
    cql_reg = compute_cql_regularization(Q, state, cql_alpha)
    Q[state, action] += alpha * (td_error - cql_reg)


# The Q-table is now trained. You can use it to derive a policy.
def select_action(state, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(A)  # Random action
    else:
        return np.argmax(Q[state])  # Action with highest Q-value


# Example usage of the trained policy
state = np.random.randint(S)
action = select_action(state)
print(f"Selected action for state {state}: {action}")


# Evaluate the learned policy
def evaluate_policy(Q, P, r, num_episodes=100, max_steps_per_episode=100):
    total_rewards = []
    for _ in range(num_episodes):
        state = np.random.randint(S)
        episode_reward = 0
        for _ in range(max_steps_per_episode):
            action = select_action(state, epsilon=0)  # Greedy action selection
            next_state = np.random.choice(S, p=P[state, action])
            reward = r[state, action]
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return total_rewards


# Evaluate the policy
rewards = evaluate_policy(Q, P, r)

# Print the results
print(f"Average return over {num_episodes} episodes: {np.mean(returns)}")
print(f"Standard deviation of return: {np.std(returns)}")
