import os
os.environ["WANDB_MODE"] = "disabled"
import wandb
from VaRFramework.src.algorithms import *
from discrete_env_utils import *
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import product
from datetime import datetime
import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Define the values you want to iterate over
models = ['machine-V2']  #,'machine-V2', 'riverswim']
seeds = [25]  #[0,1,2] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
policy_types = ['risk_averse', 'risk_seeking']
taus = [0.5]   #[0.1,0.5,0.9] #[0.1, 0.3, 0.5, 0.7, 0.9]  #[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #, 0.99]
betas = [0.01] #[0, 0.001, 0.01, 0.1, 1]
data_size_factors = [1000]    #[1000,10000] #[5,25,125] # [1, 2, 4, 8, 16, 32, 64, 128]


def plot_action_probabilities(name, models, risk_tau, beta):
    states = np.arange(models[0][1].shape[0])  # X-axis: states (same for all models)
    colors = plt.cm.viridis(np.linspace(0, 1, len(taus)))

    plt.figure(figsize=(8, 6))  # Single plot for Action 1
    plt.title(f"Replacement action tau: {risk_tau} beta: {beta}")

    for model_name, policy, _, avg, std in models:
        action_1_probs = policy[:, 1]  # Probabilities for action 1
        label_with_reward = f"{model_name} ({avg:.2f})"
        plt.plot(states, action_1_probs, label=label_with_reward, marker='o')  #, color='black')

    plt.xlabel("States")
    plt.ylabel("Probabilities")
    plt.legend()
    # plt.grid(True)
    # plt.show()
    output_dir = f"./plots/{name}/"
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    filename = f"tau{risk_tau}_beta{beta}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# Generate timestamped filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/tabular_results_{timestamp}.pkl"
policies_file = f"/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/tabular_results_policies_{timestamp}.pkl"
st_freq_file = f"/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/state_frequencies_{timestamp}.pkl"



# Initialize empty files for results and policies
with open(results_file, 'wb') as f:
    pickle.dump([], f)  # Initialize results file with an empty list
with open(policies_file, 'wb') as f:
    pickle.dump([], f)  # Initialize policies file with an empty list
with open(st_freq_file, 'wb') as f:
    pickle.dump([], f)  # Initialize policies file with an empty list

freq_actions_data = []

import os
import numpy as np
import pickle
import wandb
import matplotlib.pyplot as plt
from itertools import product
from scipy.spatial.distance import cosine
from scipy.stats import entropy

def compute_state_visitation_frequency(P, policy, initial_state_distribution, gamma=0.99):
    """
    Compute the state visitation frequency based on the whiteboard formulation.
    """
    S, A, _ = P.shape  # Number of states and actions

    # Compute Q_π (policy-induced transition matrix)
    Q_pi = np.sum(policy[:, :, np.newaxis] * P, axis=1)  # Shape (S, S)

    # Solve the linear system (I - γ Q_π^T) X = μ
    I = np.eye(S)
    A_matrix = I - gamma * Q_pi.T  # Regularized system

    try:
        X_s = np.linalg.solve(A_matrix, initial_state_distribution)  # Solve for X_s
    except np.linalg.LinAlgError:
        print("Matrix is singular, consider reducing gamma or using iterative solvers.")
        return None, None

    # Compute state-action visitation frequency: X(s, a) = X(s) * π(a | s)
    X_sa = X_s[:, np.newaxis] * policy  # Broadcasting to shape (S, A)

    return X_s, X_sa


def compute_distance(metric, freq_1, freq_2):
    """Compute distance between two state/state-action visitation distributions."""
    if metric == "cosine":
        return 1 - cosine(freq_1, freq_2)
    elif metric == "kl_divergence":
        return entropy(freq_1 + 1e-10, freq_2 + 1e-10)  # Adding small value to avoid log(0)
    else:
        return np.linalg.norm(freq_1 - freq_2, ord=2)  # L1 norm


def save_and_log_state_frequencies(method_name, state_visitation_freq, state_action_freq, st_freq_file, stored_frequencies):
    """
    Save and log state visitation frequencies for a given method.
    """
    stored_frequencies[method_name] = (state_visitation_freq, state_action_freq)

    with open(st_freq_file, 'rb+') as f:
        freq = pickle.load(f)  # Load existing frequencies
        freq.append((method_name, state_visitation_freq.tolist(), state_action_freq.tolist()))  # Append new results
        f.seek(0)  # Move to the beginning of the file
        pickle.dump(freq, f)  # Save updated results

    # Log state visitation frequencies
    wandb.log({f"{method_name}_state_frequencies": state_visitation_freq.tolist()})
    wandb.log({f"{method_name}_state_action_frequencies": state_action_freq.tolist()})

    # Print frequencies
    print(f"Method: {method_name}")
    print("State Visitation Frequencies:")
    for state, freq in enumerate(state_visitation_freq):
        print(f"State {state}: {freq}")
    print("\nState-Action Visitation Frequencies:")
    for state in range(len(state_visitation_freq)):
        for action in range(state_action_freq.shape[1]):
            print(f"State {state}, Action {action}: {state_action_freq[state, action]}")
    print("\n---\n")


import numpy as np


def compute_state_action_visitation(samples, num_states, num_actions, normalize=True, round_decimals=2):
    """
    Computes state-action visitation frequencies from offline samples.

    Args:
        samples (dict or DataFrame): Must contain 'idstatefrom' and 'idaction'.
        num_states (int): Total number of states.
        num_actions (int): Total number of actions.
        normalize (bool): Whether to normalize frequencies to probabilities.
        round_decimals (int): Number of decimals to round the normalized frequencies.

    Returns:
        np.ndarray: A (num_states x num_actions) matrix of (normalized) visitation frequencies.
    """
    # Extract transitions
    s = samples['idstatefrom']
    a = samples['idaction']

    # Initialize zero matrix
    freq_matrix = np.zeros((num_states, num_actions))

    # Count frequencies
    for si, ai in zip(s, a):
        freq_matrix[si, ai] += 1

    # Normalize if requested
    if normalize:
        total = freq_matrix.sum()
        if total > 0:
            freq_matrix = freq_matrix / total
        freq_matrix = np.round(freq_matrix, round_decimals)

    return freq_matrix

import numpy as np
import matplotlib.pyplot as plt

def plot_state_action_visitation_boxplot_grouped(samples, num_states, num_actions, normalize=True, round_decimals=2):
    """
    Computes and plots grouped state-action visitation frequencies as box plots:
    X-axis = states, Two boxes per state = actions.

    Args:
        samples (dict or DataFrame): Must contain 'idstatefrom' and 'idaction'.
        num_states (int): Total number of states.
        num_actions (int): Number of actions (should be 2 for grouped boxplot).
        normalize (bool): Normalize to probability.
        round_decimals (int): Round decimals.
    """
    assert num_actions == 2, "This plot is currently designed for 2 actions only."

    # Compute frequency matrix
    freq_matrix = np.zeros((num_states, num_actions))
    for si, ai in zip(samples['idstatefrom'], samples['idaction']):
        freq_matrix[si, ai] += 1

    if normalize:
        total = freq_matrix.sum()
        if total > 0:
            freq_matrix = freq_matrix / total
        freq_matrix = np.round(freq_matrix, round_decimals)

    # Prepare data for boxplot
    data_action_0 = [freq_matrix[s, 0] for s in range(num_states)]
    data_action_1 = [freq_matrix[s, 1] for s in range(num_states)]

    positions = np.arange(num_states)
    width = 0.35  # box width offset

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.boxplot([[v] for v in data_action_0], positions=positions - width/2, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'), medianprops=dict(color='black'))
    b2 = ax.boxplot([[v] for v in data_action_1], positions=positions + width/2, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='black'))

    ax.set_xticks(positions)
    ax.set_xticklabels([f"State {s}" for s in range(num_states)])
    ax.set_xlabel("State")
    ax.set_ylabel("Visitation Frequency")
    ax.set_title("State-Action Visitation Frequencies by State")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ['Action 0', 'Action 1'])
    ax.grid(True)

    plt.tight_layout()
    plt.show()



for model, seed, risk_tau, dsf, beta in product(models, seeds, taus, data_size_factors, betas):

    print(model, seed, risk_tau, beta, dsf)

    # Initialize wandb with the current combination of parameters
    wandb.init(project="ORL_rob", config={
        "model": model,
        "risk_tau": risk_tau,
        "beta": beta,
        "gamma": 0.9,
        "iterations": 1500,
        "num_samples": dsf,
        "delta": 0.1,
        'n_splits': 50,
        'sample_fraction': 0.5,
        'type': 'l1',
        'seed': seed
    })
    config = wandb.config

    # Initial parameters
    init_params = {
        'model': config.model,
        'seed': config.seed,
        'path': '/Users/abhilashchenreddy/PycharmProjects/CQL_AC/',
        'risk_tau': config.risk_tau,
        'num_samples': config.num_samples
    }

    data_model_params = {
        'n_splits': config.n_splits,
        'sample_fraction': config.sample_fraction
    }

    model_params = {
        'gamma': config.gamma,
        'iters': config.iterations,
        'type': config.type,
        'delta': config.delta,
        'seed': config.seed
    }

    # Environment and data gathering
    env = create_env(**init_params)

    samples_train = gather_offline_data_new(env, **init_params)

    plot_state_action_visitation_boxplot_grouped(samples_train, env.n, env.action_space.n)

    import sys
    sys.exit()

    # Extract the initial states
    idstatefrom = samples_train['idstatefrom']
    unique_states, frequencies = np.unique(idstatefrom, return_counts=True)

    # Create a dictionary with frequencies for all states, initializing missing states with 0
    all_states = np.arange(env.n)
    state_frequencies = {state: 0 for state in all_states}
    state_frequencies.update(dict(zip(unique_states, frequencies)))

    # Extract the frequencies for all states in the same order
    final_frequencies = np.array([state_frequencies[state] for state in all_states])
    # Normalize the frequencies by dividing by the sum
    if final_frequencies.sum() > 0:  # Avoid division by zero
        normalized_frequencies = final_frequencies / final_frequencies.sum()
    else:
        normalized_frequencies = final_frequencies  # All values remain 0
    normalized_frequencies = np.round(normalized_frequencies,2)

    print(normalized_frequencies)
    import sys
    sys.exit()



    # Save each policy incrementally
    with open(st_freq_file, 'rb+') as f:
        freq = pickle.load(f)  # Load existing policies
        freq.append((unique_states, frequencies))
        f.seek(0)  # Move to the beginning of the file
        pickle.dump(freq, f)  # Save updated policies

    # Log state frequencies to wandb
    state_freq_dict = {int(state): freq for state, freq in zip(unique_states, frequencies)}
    wandb.log({"state_frequencies": state_freq_dict})

    ns = env.n
    na = env.m

    data_model_params['samples'] = samples_train
    data_model_params['ns'] = ns
    data_model_params['na'] = na

    psamples, rewards, nsa = create_probability_and_reward_matrices_with_dirichlet(**data_model_params)


    # data_model_params_test = data_model_params.copy()
    # data_model_params_test['samples'] = samples_test
    # data_model_params_test['n_splits'] = 1
    # data_model_params_test['sample_fraction'] = 1
    #
    # psamples_test, rewards_test, nsa_test = create_probability_and_reward_matrices_with_dirichlet(**data_model_params_test)

    true_model, reward, initial = load_true(model=init_params['model'], path=init_params['path'])
    shape = psamples.shape
    pbar = np.mean(psamples, 0)
    true = validate_transitionp(pbar, shape[1], shape[2])

    mdp = MDP(initial, psamples.shape[1], psamples.shape[2], reward, pbar, model_params['gamma'])
    model_params['name'] = init_params['model']
    model_params['env'] = env
    model_params['mdp'] = mdp
    model_params['p_bar'] = pbar
    model_params['p_samples'] = psamples
    model_params['test_samples'] = psamples
    model_params['true_model'] = true_model
    model_params['original_data'] = samples_train
    model_params['nsa'] = nsa
    model_params['beta'] = beta
    model_params['tau'] = risk_tau

    rmdp = weighted_rmdp(**model_params)

    # Perform different value iterations and log the results
    methods = [
        # ("parametric_RobustEDAC", rmdp.parametric_robustEDAC),
        # ("RobustEDAC", rmdp.var_robust_AC_mixed_actions_robustEDAC),
        # ("EDAC", rmdp.var_robust_AC_mixed_actions_EDAC),

        # ("tabular_EDAC", rmdp.tabular_parametric_EDAC_w_parampolicy),  #new


        # ("tabular_robustEDAC", rmdp.tabular_robustEDAC),
        # ("tabular_robustEDAC_ellipsoid", rmdp.tabular_robustEDAC_ellipsoid),
        # ("run_all_three_tabular_algorithms", rmdp.run_all_three_tabular_algorithms),  #new

        # ("compare", rmdp.compare_edac_vs_robustEDAC),

        # ("asymptoticvar", rmdp.asymptotic_var_value_iteration),
        # ("var", rmdp.var_robust_value_iteration),
        # ("cvar", rmdp.cvar_robust_value_iteration),
        # ("softrobust", rmdp.soft_robust_value_iteration),
        # ("worstrobust", rmdp.worst_robust_value_iteration),
        # ("naive_bcr_l1", lambda: rmdp.compute_naive_bcr("l1")),
        # ("weighted_bcr_l1", lambda: rmdp.compute_optimized_bcr("l1", all_policies[-1][2])),
        # ("naive_bcr_linf", lambda: rmdp.compute_naive_bcr("linf")),
        # ("weighted_bcr_linf", lambda: rmdp.compute_optimized_bcr("linf", all_policies[-1][2])),
        ("true_model", rmdp.compute_true_value_new),
        ("VI_model", rmdp.value_iteration)
    ]
    stored_frequencies = {}


    for method_name, method in methods:
        if method_name in ['parametric_RobustEDAC']:
            q_network, policy_network = method()

        else:
            v, policy = method()
            print(np.round(policy, 2))

            action_1_probs = np.array(policy)[:, 1]

            combined_data = np.column_stack((normalized_frequencies, action_1_probs))

            # Add metadata as separate columns
            for row in combined_data:
                freq_actions_data.append([
                     seed, method_name,risk_tau,dsf,beta,row[0], row[1] ])

            initial_state_distribution = np.zeros(env.n)
            initial_state_distribution[0] = 1.0  # Start at initial state
            state_visitation_freq, state_action_freq = compute_state_visitation_frequency(pbar, policy,
                                                                                          initial_state_distribution)

            # Save, log, and store results
            save_and_log_state_frequencies(method_name, state_visitation_freq, state_action_freq, st_freq_file,
                                           stored_frequencies)

        train_rets = rmdp.evaluate_all(policy, mdp.rewards, method_name, test=False)
        # test_rets = rmdp.evaluate_all(policy, mdp.rewards, method_name, test=True)

        train_var = compute_value_at_risk(train_rets, model_params['delta'])

        train_mean = np.mean(train_rets)
        # test_mean = np.mean(test_rets)
        action_indices = policy
        episode_rewards = rmdp.simulate_policy(env, action_indices, render_final=False)
        sim_avg = np.mean(episode_rewards)
        sim_var = compute_value_at_risk(episode_rewards, model_params['delta'])
        print(sim_avg, sim_var)
        print("***********")

        # Append results to the results list
        result = {
            "model": model,
            "seed": seed,
            "risk_tau": risk_tau,
            "dsf": dsf,
            "beta": beta,
            "method_name": method_name,
            "train_mean": train_mean,
            "sim_mean": sim_avg
        }

        with open(results_file, 'rb+') as f:
            results = pickle.load(f)  # Load existing results
            results.append(result)
            f.seek(0)  # Move to the beginning of the file
            pickle.dump(results, f)  # Save updated results

        policy_entry = {
            "model": model,
            "seed": seed,
            "risk_tau": risk_tau,
            "beta": beta,
            "dsf": dsf,
            "method_name": method_name,
            "policy": policy
        }

        # Save the policy incrementally
        with open(policies_file, 'rb+') as f:
            policies = pickle.load(f)  # Load existing policies
            policies.append(policy_entry)  # Add new entry
            f.seek(0)  # Move to the beginning of the file
            pickle.dump(policies, f)  # Save updated policies


        # Log results to wandb
        wandb.log({
            f"{method_name}_train_var": train_var,
            f"{method_name}_train_mean": train_mean,
            f"{method_name}_sim_mean": sim_avg,
            f"{method_name}_sim_var": sim_var,
            f"{method_name}_policy": policy.tolist(),  # Log the policy
            # f"{method_name}_simulated_returns": wandb.Histogram(episode_rewards)  # Log the distribution of returns
        })

    # Finish the wandb run
    wandb.finish()
    # plot_action_probabilities(model,all_policies, risk_tau,beta)

print(stored_frequencies)
# Convert the list of arrays into a single 2D array of shape (50, 2)
combined_data_array = np.vstack(freq_actions_data)

# Create a DataFrame with appropriate column names
df = pd.DataFrame(combined_data_array)

df.to_csv("combined_data"+str(model)+".csv", index=False)

# Compute distance metrics between different methods
distances = {}
metrics =  ["l1_norm"]  #["cosine", "kl_divergence", "l1_norm"]

comparison_pairs = [
    ("tabular_EDAC", "true_model"),
    ("tabular_EDAC", "VI_model"),
    ("tabular_robustEDAC", "true_model"),
    ("tabular_robustEDAC", "VI_model")
]

for metric in metrics:
    for method_1, method_2 in comparison_pairs:
        if method_1 in stored_frequencies and method_2 in stored_frequencies:
            d_state = compute_distance(metric, stored_frequencies[method_1][0], stored_frequencies[method_2][0])
            d_state_action = compute_distance(metric, stored_frequencies[method_1][1].flatten(),
                                              stored_frequencies[method_2][1].flatten())

            key_state = f"{method_1}-{method_2}_state_{metric}"
            key_state_action = f"{method_1}-{method_2}_state_action_{metric}"

            distances[key_state] = d_state
            distances[key_state_action] = d_state_action

# Print distance results
print("\nDistance Metrics between Methods:")
for key, value in distances.items():
    print(f"{key}: {value:.4f}")
print("\n---\n")

# # Separate the independent and dependent variables
# X = combined_data_array[:, 0]  # State frequencies (independent variable)
# y = combined_data_array[:, 1]  # Action probabilities (dependent variable)
#
# # Add constant (intercept) to the independent variable matrix for statsmodels
# X = sm.add_constant(X)
#
# # Fit the linear regression model using statsmodels
# model = sm.OLS(y, X)  # OLS = Ordinary Least Squares
# results = model.fit()
#
# # Print the summary of the regression results
# print(results.summary())
#
# # Plot the regression line
# plt.scatter(X, y, label='Data points')
# plt.plot(X, results.fittedvalues, color='red', label='Regression line')
# plt.xlabel('State Frequencies')
# plt.ylabel('Action Probabilities')
# plt.legend()
# plt.show()

