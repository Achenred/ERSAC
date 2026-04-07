import os
os.environ["WANDB_MODE"] = "disabled"
import wandb
from VaRFramework.src.algorithms import *
from discrete_env_utils import *
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Define the values you want to iterate over
models =  ['riverswim']  #['machine', 'riverswim']
seeds =  [2] #[0,1,2,3,4,5,6,7,8,9]
policy_types = ['risk_neutral'] #, 'risk_averse', 'risk_seeking']

# Iterate over each combination of model, seed, and policy_type
for model in models:
    for seed in seeds:
        for policy_type in policy_types:
            # Set risk_tau based on policy_type
            if policy_type == 'risk_averse':
                risk_tau = 0.9
            elif policy_type == 'risk_seeking':
                risk_tau = 0.1
            else:  # risk_neutral
                risk_tau = 0.5

            # Initialize wandb with the current combination of parameters
            wandb.init(project="ORL_rob", config={
                "model": model,
                "seed": seed,
                "policy_type": policy_type,
                "risk_tau": risk_tau,
                "gamma": 0.9,
                "iterations": 1500,
                "num_samples": 1000,
                "delta": 0.1,
                'n_splits': 50,
                'sample_fraction': 0.5,
                'type': 'l1'
            })
            config = wandb.config

            # Initial parameters
            init_params = {
                'model': config.model,
                'seed': config.seed,
                'path': '/Users/abhilashchenreddy/PycharmProjects/CQL_AC/',
                'policy_type': config.policy_type,
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
            samples_train, samples_test = gather_offline_data(env, **init_params)

            ns = env.n
            na = env.m

            data_model_params['samples'] = samples_train
            data_model_params['ns'] = ns
            data_model_params['na'] = na

            psamples, rewards, nsa = create_probability_and_reward_matrices_with_dirichlet(**data_model_params)

            data_model_params_test = data_model_params.copy()
            data_model_params_test['samples'] = samples_test
            data_model_params_test['n_splits'] = 1
            data_model_params_test['sample_fraction'] = 1

            psamples_test, rewards_test, nsa_test = create_probability_and_reward_matrices_with_dirichlet(**data_model_params_test)

            true_model, reward, initial = load_true(model=init_params['model'], path=init_params['path'])
            shape = psamples.shape
            pbar = np.mean(psamples, 0)
            true = validate_transitionp(pbar, shape[1], shape[2])

            mdp = MDP(initial, psamples.shape[1], psamples.shape[2], reward, pbar, model_params['gamma'])
            model_params['name'] = init_params['model']
            model_params['mdp'] = mdp
            model_params['p_bar'] = pbar
            model_params['p_samples'] = psamples
            model_params['test_samples'] = psamples_test
            model_params['true_model'] = true_model
            model_params['nsa'] = nsa

            rmdp = weighted_rmdp(**model_params)

            all_policies = []

            # Perform different value iterations and log the results
            methods = [
                # ("parametric_RobustEDAC", rmdp.parametric_robustEDAC),
                ("RobustEDAC", rmdp.var_robust_AC_mixed_actions_robustEDAC),
                ("EDAC", rmdp.var_robust_AC_mixed_actions_EDAC),
                # ("asymptoticvar", rmdp.asymptotic_var_value_iteration),
                # ("var", rmdp.var_robust_value_iteration),
                # ("cvar", rmdp.cvar_robust_value_iteration),
                # ("softrobust", rmdp.soft_robust_value_iteration),
                # ("worstrobust", rmdp.worst_robust_value_iteration),
                # ("naive_bcr_l1", lambda: rmdp.compute_naive_bcr("l1")),
                # ("weighted_bcr_l1", lambda: rmdp.compute_optimized_bcr("l1", all_policies[-1][2])),
                # ("naive_bcr_linf", lambda: rmdp.compute_naive_bcr("linf")),
                # ("weighted_bcr_linf", lambda: rmdp.compute_optimized_bcr("linf", all_policies[-1][2])),
                ("true_model", rmdp.compute_true_value)
            ]

            for method_name, method in methods:
                if method_name in ['parametric_RobustEDAC']:
                    q_network, policy_network = method()
                    print(policy_network)
                    import sys
                    sys.exit()
                else:
                    v, policy = method()
                    print(policy)

                all_policies.append((method_name, policy, v))

                train_rets = rmdp.evaluate_all(policy, mdp.rewards, method_name, test=False)
                test_rets = rmdp.evaluate_all(policy, mdp.rewards, method_name, test=True)

                train_var = compute_value_at_risk(train_rets, model_params['delta'])

                train_mean = np.mean(train_rets)
                test_mean = np.mean(test_rets)
                action_indices = policy
                episode_rewards = rmdp.simulate_policy(env, action_indices, render_final=False)
                sim_avg = np.mean(episode_rewards)
                sim_var = compute_value_at_risk(episode_rewards, model_params['delta'])
                print(test_mean, sim_avg)
                print("***********")


                # Log results to wandb
                wandb.log({
                    f"{method_name}_train_var": train_var,
                    f"{method_name}_train_mean": train_mean,
                    f"{method_name}_test_mean": test_mean,
                    f"{method_name}_sim_mean": sim_avg,
                    f"{method_name}_sim_var": sim_var
                })

            # Finish the wandb run
            wandb.finish()

            # print(all_policies)
            def plot_action_probabilities(models):
                states = np.arange(models[0][1].shape[0])  # X-axis: states (same for all models)

                fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

                # Plot for Action 1
                axes[0].set_title("Action 1")
                for model_name, policy, _ in models:
                    action_1_probs = policy[:, 0]  # Probabilities for action 1
                    axes[0].plot(states, action_1_probs, label=model_name, marker='o')
                axes[0].set_xlabel("States")
                axes[0].set_ylabel("Probabilities")
                axes[0].legend()
                # axes[0].grid(True)

                # Plot for Action 2
                axes[1].set_title("Action 2")
                for model_name, policy, _ in models:
                    action_2_probs = policy[:, 1]  # Probabilities for action 2
                    axes[1].plot(states, action_2_probs, label=model_name, marker='o')  #, linestyle='--')
                axes[1].set_xlabel("States")
                axes[1].legend()
                # axes[1].grid(True)

                plt.tight_layout()
                plt.show()


            # Call the function
            plot_action_probabilities(all_policies)
