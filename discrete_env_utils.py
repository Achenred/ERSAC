import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np
from risk_agent_utils import ModelBasedRL, ModelFreeRiskRL, GridWorldMDP, GridWorldEnv, ProspectAgent, GridDisplay
import os
from sklearn.utils import resample
from scipy.stats import dirichlet

# Define the CustomEnv class
class CustomEnv(gym.Env):
    def __init__(self, df):
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
        super().reset(seed=seed)
        # Reset the state to a random state
        self.current_state = np.random.choice(self.states)
        return self.current_state

    def step(self, action):
        # Take an action and observe the next state and reward
        state_probs = self.P[self.current_state, action]
        next_state = np.random.choice(self.states, p=state_probs)
        reward = self.R[self.current_state, action, next_state]

        self.current_state = next_state

        if next_state == self.n_states -1:
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

    env = gym.make(custom_env_id, df=df)

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


def generate_samples(env, policy, num_samples, seed):
    np.random.seed(seed)
    samples = {
        'idstatefrom': [],
        'idaction': [],
        'reward': [],
        'idstateto': [],
        'step': [],
        'episode': []
    }

    for num in range(num_samples):
        state = env.reset()
        done = False


        action = policy[state]  # Get action from policy
        next_state, reward, done, _, _ = env.step(action)

        samples['idstatefrom'].append(state)
        samples['idaction'].append(action)
        samples['reward'].append(reward)
        samples['idstateto'].append(next_state)
        samples['step'].append(num)
        samples['episode'].append(0)

        # state = next_state

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
