import numpy as np
import random


class TabularActorCritic:
    def __init__(self, env, n_actions, discount=0.99, lr_actor=0.01, lr_critic=0.1):
        self.env = env
        self.n_actions = n_actions
        self.discount = discount
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Initialize value table and policy table
        self.n_states = env.observation_space.n  # Assuming discrete state space
        self.value_table = np.zeros(self.n_states)
        self.policy_table = np.full((self.n_states, n_actions), 1.0 / n_actions)

    def state_to_index(self, state):
        """
        Converts a state tuple (row, col) into a unique index.
        Adjust this function based on how your states are represented.
        """
        return state

    def choose_action(self, state):
        # Convert state to an index
        state_index = self.state_to_index(state)

        # Fetch action probabilities for the given state
        action_probs = self.policy_table[state_index]

        # Ensure the probabilities are valid (non-negative and sum to 1)
        action_probs = np.maximum(action_probs, 0)  # Ensure non-negative
        action_probs /= np.sum(action_probs)  # Normalize

        # Choose an action based on the probabilities
        action = np.random.choice(self.n_actions, p=action_probs)

        return action

    def update(self, state, action, reward, next_state, done):
        # Convert states to indices
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        # Critic update (Value function)
        td_target = reward + self.discount * self.value_table[next_state_index] * (1 - done)
        td_error = td_target - self.value_table[state_index]
        self.value_table[state_index] += self.lr_critic * td_error

        # Actor update (Policy)
        self.policy_table[state_index] *= (1 - self.lr_actor)  # Decay current policy
        self.policy_table[state_index, action] += self.lr_actor * td_error  # Update for chosen action

        # Ensure the policy remains a valid probability distribution
        self.policy_table[state_index] = np.maximum(self.policy_table[state_index], 0)  # Ensure non-negative
        self.policy_table[state_index] /= np.sum(self.policy_table[state_index])  # Normalize

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                # Update actor and critic
                self.update(state, action, reward, next_state, done)

                state = next_state


if __name__ == "__main__":
    import gym

    env = gym.make('Taxi-v3')

    agent = TabularActorCritic(env, n_actions=env.action_space.n)

    agent.train(env, num_episodes=1000)

    # Print the final policy and value tables
    print("Final Policy Table:")
    print(agent.policy_table)
    print("Final Value Table:")
    print(agent.value_table)
