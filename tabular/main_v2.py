import os
# os.environ["WANDB_MODE"] = "disabled"
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
models = ['machine-v2']  #,'machine', 'riverswim']
seeds = [5]  #[0,1,2] #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
policy_types = ['risk_averse', 'risk_seeking']
taus = [0.1]   #[0.1,0.5,0.9] #[0.1, 0.3, 0.5, 0.7, 0.9]  #[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #, 0.99]
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
        ("tabular_EDAC", rmdp.tabular_EDAC),
        ("tabular_robustEDAC", rmdp.tabular_robustEDAC),
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

    for method_name, method in methods:
        if method_name in ['parametric_RobustEDAC']:
            q_network, policy_network = method()
            print(policy_network)
            import sys

            sys.exit()
        else:
            v, policy = method()
            print(np.round(policy, 2))

            action_1_probs = np.array(policy)[:, 1]

            combined_data = np.column_stack((normalized_frequencies, action_1_probs))

            # Add metadata as separate columns
            for row in combined_data:
                freq_actions_data.append([
                     seed, method_name,risk_tau,dsf,beta,row[0], row[1] ])

            # # Combine state frequencies and action probabilities into a (10, 2) array
            # combined_data = np.column_stack((normalized_frequencies, action_1_probs))
            #
            # # Add the method name as the third column to track method-specific results
            # method_name_column = np.full((combined_data.shape[0],  method_name, seed, risk_tau, dsf),1)
            # combined_data_with_method = np.hstack((combined_data, method_name_column))
            #
            # freq_actions_data.append(combined_data_with_method)


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


# Convert the list of arrays into a single 2D array of shape (50, 2)
combined_data_array = np.vstack(freq_actions_data)

# Create a DataFrame with appropriate column names
df = pd.DataFrame(combined_data_array)

df.to_csv("combined_data"+str(model)+".csv", index=False)

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

