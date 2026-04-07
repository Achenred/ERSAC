import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np
from risk_agent_utils import ModelBasedRL, ModelFreeRiskRL, GridWorldMDP, GridWorldEnv, ProspectAgent, GridDisplay
import os
from sklearn.utils import resample
from scipy.stats import dirichlet
import random
from collections import defaultdict

# Define the CustomEnv class
class CustomEnv(gym.Env):
    def __init__(self, df,max_steps=100):
        super(CustomEnv, self).__init__()

        # Extract states, actions, transition probabilities, and rewards from the dataframe
        self.states = df['idstatefrom'].unique()
        self.actions = df['idaction'].unique()
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        # Number of states.
        self.n = self.n_states
        # Number of actions.
        self.m = self.n_actions
        self.max_steps = max_steps


        # Create a transition probability matrix and reward matrix
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))

        for _, row in df.iterrows():
            state_from = int(row['idstatefrom'])
            action = int(row['idaction'])
            state_to = int(row['idstateto'])
            prob = row['probability']
            reward = row['reward']

            self.P[state_from, action, state_to] = prob
            self.R[state_from, action, state_to] = reward

        # Define the action and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Discrete(self.n_states)

        # Initialize the state
        self.current_state = None

        n_states, n_actions, _ = self.P.shape

        for state_from in range(n_states):
            for action in range(n_actions):
                # Sum the transition probabilities for this state-action pair
                total_prob = np.sum(self.P[state_from, action])
                if total_prob > 0:
                    # Normalize the probabilities
                    self.P[state_from, action] /= total_prob
                else:
                    pass


    def reset(self, seed=None, options=None):
        # super().reset() #seed=None)  #seed)
        # Reset the state to a random state

        self.current_state = np.random.choice(self.states)
        self.step_count = 0  # Reset the step count
        return self.current_state

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     # Ensure randomness for the initial state
    #     rng = np.random.default_rng(seed)  # Create a random number generator with the given seed
    #     self.current_state = rng.choice(self.states)  # Use the random number generator for sampling
    #     return self.current_state

    def step(self, action):
        # Take an action and observe the next state and reward
        state_probs = self.P[self.current_state, action]
        next_state = np.random.choice(self.states, p=state_probs)
        reward = self.R[self.current_state, action, next_state]

        self.current_state = next_state
        self.step_count += 1

        if next_state == self.n_states-1 or self.step_count >= self.max_steps:
            done = True
        else:
            done = False  # Define your own terminal condition if needed

        return next_state, reward, done, {}, {}

    def render(self, mode='human'):
        # Implement rendering logic if needed
        pass

def create_env(**kwargs):
    # Load your CSV file
    path = kwargs.get('path')
    model = kwargs.get('model')

    csv_file = str(path)+"discrete_domains/domains/"+str(model)+".csv"
    df = pd.read_csv(csv_file)

    # Select the columns to reduce by 1
    columns_to_reduce = ['idstatefrom', 'idaction', 'idstateto']
    df[columns_to_reduce] = df[columns_to_reduce] - 1

    custom_env_id = str(model)+'Env-v0'
    env_ids = [env_spec.id for env_spec in registry.values()]

    if custom_env_id not in env_ids:
        # Register the custom environment
        register(
            id=custom_env_id,
            entry_point=CustomEnv,  # Directly use the class name
            max_episode_steps=1000,
        )

        print(f"{custom_env_id} is successfully registered.")
    else:
        print(f"{custom_env_id} is already registered.")

    env = gym.make(custom_env_id, df=df, max_steps = 100)

    return env

def generate_batch_data(env, policy):

    # Test the environment
    state, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Replace with your action selection logic
        next_state, reward, done, _, _ = env.step(action)
        env.render()

        print(reward)

    env.close()


def generate_samples(env, policy, num_samples, seed, reset_frequency=10):
    """
    Generates samples from the environment by following a policy.

    Parameters:
        env: The environment.
        policy: A dictionary or function that maps states to actions.
        num_samples: The total number of samples to generate.
        seed: Random seed for reproducibility (optional).
        reset_frequency: Frequency of environment resets (default is 1).

    Returns:
        A dictionary containing the collected samples.
    """
    # Set the random seed for reproducibility, if needed

    if seed is not None:
        np.random.seed(seed)

    samples = {
        'idstatefrom': [],
        'idaction': [],
        'reward': [],
        'idstateto': [],
        'step': [],
        'episode': []
    }

    total_samples = 0  # Counter for the total number of samples collected
    episode_count = 0  # Counter for episode numbers

    init_states = []

    ns = env.n  # Number of states
    na = env.action_space.n  # Number of actions
    num_samples = num_samples * ns

    state = env.reset()  # Initialize state
    step_in_episode = 0
    ns = env.n
    num_samples = num_samples*ns

    while total_samples < num_samples:
        # If the episode naturally ends (done) or reset frequency is reached, reset the environment
        if step_in_episode % reset_frequency == 0 or step_in_episode == 0 or done:
            state = env.reset()
            episode_count += 1  # Increment the episode counter
            init_states.append(state)
            step_in_episode = 0  # Reset step counter for new episode

        if random.random() < 0.1:  # 10% probability
            action = env.action_space.sample()  # Sample a random action

            next_state, reward, done, _, _ = env.step(action)

            # Save the transition
            samples['idstatefrom'].append(state)
            samples['idaction'].append(action)
            samples['reward'].append(reward)
            samples['idstateto'].append(next_state)
            samples['step'].append(step_in_episode)
            samples['episode'].append(episode_count)

            # Update counters
            state = next_state
            step_in_episode += 1
            total_samples += 1


        else:
            action = policy[state]  # Get the action based on the policy

            next_state, reward, done, _, _ = env.step(action)

            # Save the transition
            samples['idstatefrom'].append(state)
            samples['idaction'].append(action)
            samples['reward'].append(reward)
            samples['idstateto'].append(next_state)
            samples['step'].append(step_in_episode)
            samples['episode'].append(episode_count)

            # Update counters
            state = next_state
            step_in_episode += 1
            total_samples += 1


        # Break if we reach the desired number of samples
        if total_samples >= num_samples:
            break

    unique_states, frequencies = np.unique(init_states, return_counts=True)
    print(frequencies)

    # Initialize transition frequency dictionary
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    action_counts = defaultdict(lambda: defaultdict(int))

    # Count transitions from the collected samples
    for i in range(len(samples['idstatefrom'])):
        state = samples['idstatefrom'][i]
        action = samples['idaction'][i]
        next_state = samples['idstateto'][i]

        transition_counts[state][action][next_state] += 1
        action_counts[state][action] += 1

    print("idstatefrom,idaction,idstateto,count")
    for state in sorted(transition_counts.keys()):
        for action in sorted(transition_counts[state].keys()):
            for next_state in sorted(transition_counts[state][action].keys()):
                count = transition_counts[state][action][next_state]
                # print(f"{state},{action},{next_state},{count}")

    # Print transition probabilities
    print("idstatefrom,idaction,idstateto,probability")
    for state in sorted(transition_counts.keys()):
        for action in sorted(transition_counts[state].keys()):
            total_action_count = action_counts[state][action]
            for next_state in sorted(transition_counts[state][action].keys()):
                probability = np.round(transition_counts[state][action][next_state] / total_action_count,2)
                print(f"{state},{action},{next_state},{probability:.2f}")


    return samples

def save_samples(samples, model, seed, path,policy_type, train = True):
    # Create directories if they do not exist
    directory = f"{path}/offline_data/{policy_type}/{model}/{seed}/"
    os.makedirs(directory, exist_ok=True)

    # Define the file path
    if train == True:
        file_path = os.path.join(directory, "samples_train.npz")
    else:
        file_path = os.path.join(directory, "samples_test.npz")

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Samples already exist in {file_path}. No new data is saved.")
        return

    # Save the samples as a NumPy .npz file
    np.savez(file_path,
             idstatefrom =np.array(samples['idstatefrom']),
             idaction =np.array(samples['idaction']),
             reward =np.array(samples['reward']),
             idstateto =np.array(samples['idstateto']),
             episode =np.array(samples['episode']),
             step =np.array(samples['step']))
    print(f"Samples saved in {file_path}.")


def load_samples(model,seed,path,policy_type, train = True):

    # Define the file path
    if train == True:
        file_path = f"{path}/offline_data/{policy_type}/{model}/{seed}/samples_train.npz"
        status = True
    else:
        file_path = f"{path}/offline_data/{policy_type}/{model}/{seed}/samples_test.npz"
        status = True


    if not os.path.exists(file_path):
        status = False
        print(f"No samples found at {file_path}")
        return None, status

    # Load the samples from the .npz file
    data = np.load(file_path)
    print(data.files)

    # Convert the loaded data to a dictionary
    samples = {key: data[key] for key in data.files}

    print(f"Samples loaded from {file_path}.")
    return samples, status

def gather_offline_data(env, **kwargs):
    model = kwargs.get('model')
    seed = kwargs.get('seed')
    path = kwargs.get('path')
    num_samples = kwargs.get('num_samples')
    policy_type = kwargs.get('policy_type')
    risk_tau = kwargs.get('risk_tau')

    samples_train, status_train = load_samples(model, seed,path,policy_type, train = True)
    if policy_type == 'risk_neutral':
        model_based_rl = ModelBasedRL()
        model_based_rl.policy_iteration(env)
        policy = model_based_rl.policy
        # model_based_rl.simulate_policy(env,num_episodes = 1000,  render_final=False)

    elif policy_type == 'risk_averse':

        model_based_rl = ModelBasedRL()
        model_based_rl.dynamic_risk_model(env, risk_tau=risk_tau)
        policy = model_based_rl.policy


    elif policy_type == 'risk_seeking':
        model_based_rl = ModelBasedRL()
        model_based_rl.dynamic_risk_model(env, risk_tau=risk_tau)
        policy = model_based_rl.policy



    if status_train == False:
        samples_train = generate_samples(env, policy, num_samples, seed)
        save_samples(samples_train, model, seed,path,policy_type, train = True)

    samples_test, status_test = load_samples(model, seed, path,policy_type, train=False)
    if status_test == False:
        samples_test = generate_samples(env, policy, int(num_samples*0.2), seed)
        save_samples(samples_test, model, seed,path,policy_type, train = False)

    return samples_train, samples_test

def load_samples_new(model, seed, path, risk_tau,num_samples, train=True):
    """
    Load or check the existence of training/testing samples for a given model and seed.
    """
    # Define the file path
    file_suffix = "samples_train.npz" if train else "samples_test.npz"
    file_path = f"{path}offline_data/{model}/{seed}/{risk_tau}/{num_samples}/{file_suffix}"


    if not os.path.exists(file_path):
        print(f"No samples found at {file_path}")
        return None, False

    # Load the samples from the .npz file
    data = np.load(file_path)
    print(data.files)

    # Convert the loaded data to a dictionary
    samples = {key: data[key] for key in data.files}

    print(f"Samples loaded from {file_path}.")
    return samples, True

def save_samples_new(samples, model, seed, path,risk_tau,num_samples, train = True):
    # Create directories if they do not exist
    directory = f"{path}offline_data/{model}/{seed}/{risk_tau}/{num_samples}"
    os.makedirs(directory, exist_ok=True)

    # Define the file path
    if train == True:
        file_path = os.path.join(directory, "samples_train.npz")
    else:
        file_path = os.path.join(directory, "samples_test.npz")

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Samples already exist in {file_path}. No new data is saved.")
        return

    # Save the samples as a NumPy .npz file
    np.savez(file_path,
             idstatefrom =np.array(samples['idstatefrom']),
             idaction =np.array(samples['idaction']),
             reward =np.array(samples['reward']),
             idstateto =np.array(samples['idstateto']),
             episode =np.array(samples['episode']),
             step =np.array(samples['step']))
    print(f"Samples saved in {file_path}.")


def gather_offline_data_new(env, **kwargs):
    """
    Load or generate offline data for training and testing.
    """
    model = kwargs.get('model')
    seed = kwargs.get('seed')
    path = kwargs.get('path')
    num_samples = kwargs.get('num_samples')
    policy_type = kwargs.get('policy_type')
    risk_tau = kwargs.get('risk_tau')

    # Load or simulate training data
    samples_train, status_train = load_samples_new(model, seed, path, risk_tau,num_samples, train=True)
    if not status_train:
        if risk_tau == 0.5:

            model_based_rl = ModelBasedRL()
            model_based_rl.q_value_iteration(env)
            policy = model_based_rl.policy

        else:
            model_based_rl = ModelBasedRL()
            model_based_rl.dynamic_risk_model(env, risk_tau=risk_tau)
            policy = model_based_rl.policy

        # Generate and save training samples
        samples_train = generate_samples(env, policy, num_samples, seed)
        save_samples_new(samples_train, model, seed, path, risk_tau,num_samples, train=True)

    # # Load or simulate testing data
    # samples_test, status_test = load_samples_new(model, seed, path, policy_type, train=False)
    # if not status_test:
    #     if 'policy' not in locals():  # Ensure policy is defined
    #         if risk_tau == 0.5:
    #             model_based_rl = ModelBasedRL()
    #             model_based_rl.policy_iteration(env)
    #             policy = model_based_rl.policy
    #         else:
    #             model_based_rl = ModelBasedRL()
    #             model_based_rl.dynamic_risk_model(env, risk_tau=risk_tau)
    #             policy = model_based_rl.policy
    #
    #     # Generate and save testing samples
    #     samples_test = generate_samples(env, policy, int(num_samples * 0.2), seed)
    #     save_samples(samples_test, model, seed, path, policy_type, train=False)

    return samples_train


def create_probability_and_reward_matrices(**data_model_params):
    samples = data_model_params.get('samples')
    ns = data_model_params.get('ns')
    na = data_model_params.get('na')
    n_splits = data_model_params.get('n_splits')
    sample_fraction = data_model_params.get('sample_fraction')

    def load_nsa(samples, ns, na):
        # Extract columns from samples
        idstatefrom = samples['idstatefrom']
        idaction = samples['idaction']
        idstateto = samples['idstateto']

        nsa = np.zeros((ns, na, ns))
        lent = len(idstatefrom)

        for idx in range(lent):
            s = idstatefrom[idx]
            a = idaction[idx]
            ns_ = idstateto[idx]
            nsa[s, a, ns_] += 1

        return nsa

    idstatefrom = samples['idstatefrom']
    idaction = samples['idaction']
    idstateto = samples['idstateto']
    reward = samples['reward']

    no = n_splits  # Number of different samples, treated as 'idoutcome'

    psamples = np.zeros((no, ns, na, ns))
    rewards = np.zeros((no, ns, na, ns))
    counts = np.zeros((no, ns, na, ns))  # To count occurrences for averaging rewards

    for i in range(n_splits):
        # Generate a bootstrap sample (80% of data)
        indices = resample(range(len(idstatefrom)), replace=False, n_samples=int(sample_fraction * len(idstatefrom)))

        for idx in indices:
            s = idstatefrom[idx]
            a = idaction[idx]
            s_ = idstateto[idx]

            psamples[i, s, a, s_] += 1  # Count occurrences for probability
            rewards[i, s, a, s_] += reward[idx]  # Sum rewards
            counts[i, s, a, s_] += 1  # Increment count for averaging

        # Normalize the probabilities for the current outcome/sample
        psamples[i] = psamples[i] / np.maximum(1, np.sum(psamples[i], axis=-1, keepdims=True))
        # Average the rewards
        rewards[i] = np.divide(rewards[i], np.maximum(1, counts[i]))

    nsa = load_nsa(samples, ns, na)

    return psamples, rewards, nsa


def create_probability_and_reward_matrices_with_dirichlet(**data_model_params):
    samples = data_model_params.get('samples')
    ns = data_model_params.get('ns')
    na = data_model_params.get('na')
    n_splits = data_model_params.get('n_splits')
    sample_fraction = data_model_params.get('sample_fraction')
    alpha = data_model_params.get('alpha', 1)  # Dirichlet smoothing parameter

    def load_nsa(samples, ns, na):
        # Extract columns from samples
        idstatefrom = samples['idstatefrom']
        idaction = samples['idaction']
        idstateto = samples['idstateto']

        nsa = np.zeros((ns, na, ns))
        lent = len(idstatefrom)

        for idx in range(lent):
            s = idstatefrom[idx]
            a = idaction[idx]
            ns_ = idstateto[idx]
            nsa[s, a, ns_] += 1

        return nsa

    idstatefrom = samples['idstatefrom']
    idaction = samples['idaction']
    idstateto = samples['idstateto']
    reward = samples['reward']

    no = n_splits  # Number of different samples, treated as 'idoutcome'

    psamples = np.zeros((no, ns, na, ns))
    rewards = np.zeros((no, ns, na, ns))
    counts = np.zeros((no, ns, na, ns))  # To count occurrences for averaging rewards

    for i in range(n_splits):
        # Generate a bootstrap sample (80% of data)
        indices = resample(range(len(idstatefrom)), replace=False, n_samples=int(sample_fraction * len(idstatefrom)))

        for idx in indices:
            s = idstatefrom[idx]
            a = idaction[idx]
            s_ = idstateto[idx]

            counts[i, s, a, s_] += 1  # Increment count for Dirichlet
            rewards[i, s, a, s_] += reward[idx]  # Sum rewards

        # Generate psamples using Dirichlet distribution
        for s in range(ns):
            for a in range(na):
                dirichlet_params = counts[i, s, a, :] + alpha
                # psamples[i, s, a, :] = dirichlet.rvs(dirichlet_params, size=1, random_state=1)# dirichlet.mean(dirichlet_params)   #np.random.dirichlet(dirichlet_params)
                psamples[i, s, a, :] = dirichlet.mean(dirichlet_params)

        # Average the rewards
        rewards[i] = np.divide(rewards[i], np.maximum(1, counts[i]))

    nsa = load_nsa(samples, ns, na)

    return psamples, rewards, nsa

def compute_value_at_risk(returns, confidence_level=0.95):
    """
    Computes the Value at Risk (VaR) for a list of returns.

    :param returns: List or array of returns.
    :param confidence_level: Confidence level for VaR (default is 0.95 for 95% confidence).
    :return: Value at Risk (VaR).
    """
    # Sort returns in ascending order
    sorted_returns = np.sort(returns)

    # Calculate the index corresponding to the confidence level
    index = int((1 - confidence_level) * len(sorted_returns))

    # VaR is the return at this index
    var = sorted_returns[index]

    return var



if __name__=='__main__':
    # Generate samples
    num_samples = 10  # Number of samples you want to generate
    seed =0
    model = 'machine'
    path = '/Users/abhilashchenreddy/PycharmProjects/CQL_AC/'
    policy_type = "risk neutral"

    env = create_env(model, path)
    samples_train, samples_test = gather_offline_data(model,seed,path, env, policy_type)
