import numpy as np

from collections import defaultdict
from datetime import datetime
import pandas as pd

def expectile_risk(tau):
    def u_f(y):
        return 2 * ((1 - tau) * np.maximum(0, y) - tau * np.maximum(0, -y))

    def rho_int(p, v, u_f):
        tol = np.sqrt(np.finfo(float).eps)
        if np.allclose(v, v[0]):  # handle degenerate case
            return v[0]
        bounds = [np.min(v), np.max(v)]
        while bounds[1] - bounds[0] > tol:
            mid = np.mean(bounds)
            tmp = np.dot(p, u_f(v - mid))
            if tmp >= 0:
                bounds[0] = mid
            else:
                bounds[1] = mid
        return np.mean(bounds)

    def rho_f(p, v):
        return rho_int(p, v, u_f)

    return rho_f





def get_trajectories(gamma, S, A, P, r, policy, M):
    reset_prob = 1 - gamma
    random_a_prob = 0.2
    trajs = []
    s = np.random.randint(S)
    for _ in range(M):
        if np.random.rand() < reset_prob:
            s = np.random.randint(S)
        if np.random.rand() < random_a_prob:
            a = np.random.randint(A)
        else:
            a = np.random.choice(np.where(policy[s, :] == 1)[0])
        sprime = np.random.choice(S, p=P[s, a, :] / P[s, a, :].sum())
        trajs.append([s, a, sprime, r[s, a,sprime]])
        s = sprime
    return np.array(trajs)



def get_state_action_visitation(trajs, S, A):
    freq = np.zeros((S, A), dtype=int)
    for s, a, _, _ in trajs:
        freq[int(s), int(a)] += 1
    return freq





# def load_environment(data):
#     states = set()
#     actions = set()
#     for entry in data:
#         states.add(entry[0])
#         actions.add(entry[1])
#         states.add(entry[2])
#     S, A = max(states), max(actions)
#     P = np.zeros((S, A, S))
#     r = np.zeros((S, A))
#     for s, a, sprime, prob, rew in data:
#
#         P[s - 1, a - 1, sprime - 1] = prob
#         # r[s - 1, a - 1] += prob * rew
#         r[s - 1, a - 1]  =  rew
#         # r[s-1,a-1,sprime-1] = rew
#     return S, A, P, r

def load_environment(data):
    states = set()
    actions = set()
    for entry in data:
        states.add(entry[0])
        actions.add(entry[1])
        states.add(entry[2])
    S, A = max(states), max(actions)
    P = np.zeros((S, A, S))
    r = np.zeros((S, A, S))
    for s, a, sprime, prob, rew in data:
        P[s - 1, a - 1, sprime - 1] = prob
        # r[s - 1, a - 1] += prob * rew
        r[s - 1, a - 1, sprime - 1] = rew
        # r[s-1,a-1,sprime-1] = rew
    return S, A, P, r


def vi_expectile(risk_tau, gamma, S, A, P, r, max_iter=10000, tol=1e-6):
    def expectile_risk(tau):
        def u_f(y):
            return 2 * ((1 - tau) * np.maximum(0, y) - tau * np.maximum(0, -y))

        def rho_int(p, v, u_f):
            tol = np.sqrt(np.finfo(float).eps)
            bounds = [np.min(v), np.max(v)]
            while bounds[1] - bounds[0] > tol:
                mid = np.mean(bounds)
                tmp = np.dot(p, u_f(v - mid))
                if tmp >= 0:
                    bounds[0] = mid
                else:
                    bounds[1] = mid
            return np.mean(bounds)

        def rho_f(p, v):
            return rho_int(p, v, u_f)

        return rho_f, u_f

    rho_f, u_f = expectile_risk(risk_tau)

    Q = np.zeros((S, A))
    Q_ = np.zeros((S, A))
    hs = []

    for it in range(max_iter):
        V = np.max(Q, axis=1)  # value function
        for s in range(S):
            for a in range(A):
                rewards = r[s, a, :]  # shape (n,)
                transitions = P[s, a, :]  # shape (n,)
                Q_sa_vector = rewards + gamma * V  # total return vector
                Q_[s, a] = rho_f(transitions, Q_sa_vector)

    # for _ in range(max_iter):
    #
    #     V = np.max(Q, axis=1)  # value function
    #     for s in range(S):
    #         for a in range(A):
    #             Q_[s, a] = rho_f(P[s, a, :], r[s, a] + gamma * V)

        hs.append(np.linalg.norm(Q - Q_))

        if np.linalg.norm(Q - Q_) / np.sqrt(S * A) < tol:
            print(it)
            break
        Q = Q_.copy()

    # Return Q and one-hot policy
    policy = np.zeros((S, A))
    policy[np.arange(S), np.argmax(Q, axis=1)] = 1
    return Q, policy

def value_iteration(gamma, S, A, P, r, max_iter=10000, tol=1e-6):
    """
    Standard value iteration for risk-neutral expected reward.
    """
    V = np.zeros(S)
    for _ in range(max_iter):
        V_new = np.zeros(S)
        for s in range(S):
            Q_sa = np.zeros(A)
            for a in range(A):
                Q_sa[a] = np.sum(P[s, a, :] * (r[s, a, :] + gamma * V))
            V_new[s] = np.max(Q_sa)
        if np.linalg.norm(V - V_new) < tol:
            break
        V = V_new

    # Extract policy
    policy = np.zeros((S, A))
    for s in range(S):
        Q_sa = np.zeros(A)
        for a in range(A):
            Q_sa[a] = np.sum(P[s, a, :] * (r[s, a, :] + gamma * V))
        policy[s, np.argmax(Q_sa)] = 1.0
    return V, policy




def run_sacn(gamma, S, A, trajs, ensN, batch_size=32, tol=1e-6, max_iter=5000):
    Qs = np.random.rand(S, A, ensN) / (1 - gamma)
    step_count = 1
    step_size = lambda _: 0.1
    prev_Qs = np.copy(Qs)
    for it in range(max_iter):

        batch_indices = np.random.choice(len(trajs), size=min(batch_size, len(trajs)), replace=False)
        batch = trajs[batch_indices]
        for s, a, sprime, rsa in batch:
            min_Q = np.min(Qs, axis=2)
            y = rsa + gamma * np.max(min_Q[int(sprime), :])
            Qs[int(s), int(a), :] += step_size(step_count) * (y - Qs[int(s), int(a), :])
            step_count += 1
        if np.linalg.norm(Qs - prev_Qs) < tol:
            break
        prev_Qs = np.copy(Qs)
    Q_star = np.min(Qs, axis=2)
    policy = (Q_star == Q_star.max(axis=1, keepdims=True)).astype(float)
    # def softmax(x, axis=1, tau=100.0):
    #     x = x / tau
    #     x_max = np.max(x, axis=axis, keepdims=True)
    #     e_x = np.exp(x - x_max)
    #     return e_x / np.sum(e_x, axis=axis, keepdims=True)
    #
    # policy = np.round(softmax(Q_star, axis=1, tau=0.1),2)  # use small tau for sharper policy

    print(Q_star)
    print(policy)
    import sys
    sys.exit()

    return Q_star, policy

def softmax(q_values, tau=1.0):
    q_values = np.asarray(q_values) / tau

    if q_values.ndim == 1:
        q_max = np.max(q_values)
        e_q = np.exp(q_values - q_max)
        return e_q / np.sum(e_q)
    elif q_values.ndim == 2:
        q_max = np.max(q_values, axis=1, keepdims=True)
        e_q = np.exp(q_values - q_max)
        return e_q / np.sum(e_q, axis=1, keepdims=True)
    else:
        raise ValueError("Input to softmax must be 1D or 2D.")



def run_sacn_box(gamma, alpha, S, A, trajs, ensN, batch_size=32,
                 eta_q=0.1, eta_pi=0.1, tau=0.1, max_iter=10000, tol=1e-5,
                 verbose=True):
    Qs = np.random.rand(S, A, ensN) / (1 - gamma)
    Qs_target = np.copy(Qs)
    policy_logits = np.zeros((S, A))

    for it in range(max_iter):
        old_Qs = Qs.copy()
        old_logits = policy_logits.copy()

        batch_indices = np.random.choice(len(trajs), size=min(batch_size, len(trajs)), replace=False)
        batch = trajs[batch_indices]

        for s, a, sprime, r in batch:
            s, a, sprime = int(s), int(a), int(sprime)

            # Conservative Q for next state (min across ensemble)
            q_next = np.min(Qs_target[sprime], axis=1)
            pi_sprime = softmax(policy_logits[sprime])
            entropy = -np.sum(pi_sprime * np.log(pi_sprime + 1e-8))
            v_sprime = np.sum(pi_sprime * q_next)
            y = r + gamma * (v_sprime - alpha * entropy)

            # Critic update
            for i in range(ensN):
                Qs[s, a, i] += eta_q * (y - Qs[s, a, i])

            # Actor update
            min_q_s = np.min(Qs[s], axis=1)
            pi_s = softmax(policy_logits[s])
            advantage = min_q_s - np.dot(pi_s, min_q_s)
            grad_log_pi = pi_s * advantage
            policy_logits[s] += eta_pi * (grad_log_pi - alpha * np.log(pi_s + 1e-8))

        # Target update (Polyak averaging)
        Qs_target = tau * Qs + (1 - tau) * Qs_target

        # Diagnostics
        delta_q = np.linalg.norm(Qs - old_Qs)
        delta_pi = np.linalg.norm(policy_logits - old_logits)

        if verbose and it % 100 == 0:
            entropy_avg = np.mean(
                [-np.sum(softmax(policy_logits[s]) * np.log(softmax(policy_logits[s]) + 1e-8)) for s in range(S)])
            print(f"Iter {it:4d} | ΔQ: {delta_q:.4f} | Δπ: {delta_pi:.4f} | Avg Entropy: {entropy_avg:.4f}")

        if delta_q < tol and delta_pi < tol:
            if verbose:
                print(f"Converged at iteration {it}")
            break

    Q_star = np.min(Qs, axis=2)
    final_policy = np.vstack([softmax(policy_logits[s]) for s in range(S)])

    print(Q_star)
    print(final_policy)
    import sys
    sys.exit()

    return Q_star, final_policy

def run_sacn_convex_hull(gamma, alpha, S, A, trajs, ensN, batch_size=32,
                         eta_q=0.1, eta_pi=0.1, tau=0.005, max_iter=1000, tol=1e-6):
    Qs = np.random.rand(S, A, ensN) / (1 - gamma)
    Qs_target = np.copy(Qs)
    policy_logits = np.zeros((S, A))

    def softmax(q_values, tau=1.0):
        q_values = q_values / tau
        q_max = np.max(q_values, axis=1, keepdims=True)
        e_q = np.exp(q_values - q_max)
        return e_q / np.sum(e_q, axis=1, keepdims=True)

    for it in range(max_iter):
        batch_indices = np.random.choice(len(trajs), size=min(batch_size, len(trajs)), replace=False)
        batch = trajs[batch_indices]

        for s, a, sprime, r in batch:
            s, a, sprime = int(s), int(a), int(sprime)

            q_ensemble_next = Qs_target[sprime].T  # (ensN, A)
            pi_sprime = softmax(policy_logits[sprime][None, :])[0]
            entropy = -np.sum(pi_sprime * np.log(pi_sprime + 1e-8))

            values = np.dot(q_ensemble_next, pi_sprime)  # (ensN,)
            i_star = np.argmin(values)
            v_sprime = np.sum(q_ensemble_next[i_star] * pi_sprime)

            y = r + gamma * (v_sprime - alpha * entropy)

            for i in range(ensN):
                Qs[s, a, i] += eta_q * (y - Qs[s, a, i])

            q_selected = Qs[s, :, i_star]
            pi_s = softmax(policy_logits[s][None, :])[0]
            grad_log_pi = pi_s * (q_selected - np.sum(q_selected * pi_s))
            policy_logits[s] += eta_pi * (grad_log_pi - alpha * np.log(pi_s + 1e-8))

        Qs_target = tau * Qs + (1 - tau) * Qs_target

        if np.linalg.norm(Qs - Qs_target) < tol:
            break

    Q_star = np.min(Qs, axis=2)
    final_policy = softmax(policy_logits)

    return Q_star, final_policy


def run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs, ensN, quantile=0.95, batch_size=32,
                       eta_q=0.1, eta_pi=0.1, tau=0.005, max_iter=1000, tol=1e-6):

    def softmax(x, tau=1.0):
        x = x / tau
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    Qs = np.random.rand(S, A, ensN) / (1 - gamma)
    Qs_target = np.copy(Qs)
    policy_logits = np.zeros((S, A))

    for it in range(max_iter):
        batch_indices = np.random.choice(len(trajs), size=min(batch_size, len(trajs)), replace=False)
        batch = trajs[batch_indices]

        for s, a, sprime, r in batch:
            s, a, sprime = int(s), int(a), int(sprime)

            # Get Q samples at s'
            Q_samples = Qs_target[sprime]  # shape [A, ensN]
            q_mean = Q_samples.mean(axis=1)  # [A]
            q_centered = Q_samples - q_mean[:, None]  # [A, ensN]
            cov = np.dot(q_centered, q_centered.T) / (ensN - 1)  # [A, A]


            pi_sprime = softmax(policy_logits[sprime][None, :])[0]  # [A]
            entropy = -np.sum(pi_sprime * np.log(pi_sprime + 1e-8))

            # Compute Mahalanobis distances and quantile-based radius
            reg_cov = cov + 1e-6 * np.eye(A)
            inv_cov = np.linalg.inv(reg_cov)


            # Compute Mahalanobis distances in vectorized form
            mahalanobis = np.einsum('bi,ij,bj->b', q_centered.T, inv_cov, q_centered.T)

            # mahalanobis = [
            #     np.dot((Q_samples[:, i] - q_mean).T, np.linalg.solve(cov + 1e-6 * np.eye(A), Q_samples[:, i] - q_mean))
            #     for i in range(ensN)
            # ]

            radius = np.sqrt(np.quantile(mahalanobis, quantile))

            # Compute robust value
            cov_pi = np.dot(cov, pi_sprime)
            penalty = radius * np.linalg.norm(cov_pi)
            v_sprime = np.sum(pi_sprime * q_mean) -  penalty

            # Bellman target with entropy regularization
            y = r + gamma * (v_sprime - beta * entropy)

            # Critic update
            for i in range(ensN):
                Qs[s, a, i] += eta_q * (y - Qs[s, a, i])

            # Actor update using the robust gradient
            Q_samples_s = Qs[s]  # [A, ensN]
            q_mean_s = Q_samples_s.mean(axis=1)
            q_centered_s = Q_samples_s - q_mean_s[:, None]
            cov_s = np.dot(q_centered_s, q_centered_s.T) / (ensN - 1)
            pi_s = softmax(policy_logits[s][None, :])[0]

            # mahalanobis_s = [
            #     np.dot((Q_samples_s[:, i] - q_mean_s).T,
            #            np.linalg.solve(cov_s + 1e-6 * np.eye(A), Q_samples_s[:, i] - q_mean_s))
            #     for i in range(ensN)
            # ]

            reg_cov_s = cov_s + 1e-6 * np.eye(A)
            inv_cov_s = np.linalg.inv(reg_cov_s)
            # Compute Mahalanobis distances in vectorized form
            mahalanobis_s = np.einsum('bi,ij,bj->b', q_centered_s.T, inv_cov_s, q_centered_s.T)
            radius_s = np.sqrt(np.quantile(mahalanobis_s, quantile))

            cov_pi_s = np.dot(cov_s, pi_s)
            penalty_s = radius_s * cov_pi_s / (np.linalg.norm(cov_pi_s) + 1e-8)

            grad_log_pi = q_mean_s - penalty_s - alpha * np.log(pi_s + 1e-8)
            grad_log_pi -= np.sum(grad_log_pi * pi_s)  # REINFORCE baseline
            policy_logits[s] += eta_pi * grad_log_pi

        Qs_target = tau * Qs + (1 - tau) * Qs_target

        if np.linalg.norm(Qs - Qs_target) < tol:
            break

    Q_star = np.min(Qs, axis=2)
    final_policy = softmax(policy_logits)
    return Q_star, final_policy

def evaluate_policy(policy, P, r, S, A, gamma=0.9, num_episodes=1000, max_steps=100):

    total_rewards = []

    for n in range(num_episodes):
        s = np.random.randint(S)
        total_reward = 0.0
        discount = 1.0
        steps_taken = 0

        for _ in range(max_steps):
            a = np.random.choice(A, p=policy[s])

            sprime = np.random.choice(S, p=P[s, a] / P[s, a].sum())

            reward = r[s, a, sprime]
            total_reward += discount * reward
            # discount *= gamma
            steps_taken += 1
            print(_,s, a, sprime, reward)

            if s == S-1:
                break
            s = sprime
        normalized_return = total_reward / steps_taken if steps_taken > 0 else 0.0
        print("**")
        print(_, normalized_return)

        total_rewards.append(normalized_return)
    print(np.std(total_rewards))
    return np.mean(total_rewards)

def evaluate_policy_normalized(policy_eval, policy_random, policy_optimal, P, r, S, A, gamma=0.9, num_episodes=1000, max_steps=100):
    def run_policy(policy):
        total_returns = []
        for _ in range(num_episodes):
            print("episode: "+str(_))
            s = 0  #np.random.randint(S)
            total_reward = 0.0
            discount = 1.0
            for steps in range(max_steps):
                a = np.random.choice(A, p=policy[s])
                # a = int(np.argmax(policy[s]))
                # print(a,policy[s])
                sprime = np.random.choice(S, p=P[s, a] / P[s, a].sum())
                reward = r[s, a, sprime]
                total_reward += discount * reward
                discount *= gamma
                # if (s == S - 1) :
                #     break
                # if a == 1:
                #     break
                print(s, a, sprime, reward, policy[s], total_reward)
                s = sprime

            total_returns.append(total_reward)
        return np.mean(total_returns), np.std(total_returns)

    # Run all policies
    print("eval policy")
    mean_eval, std_eval = run_policy(policy_eval)
    print("random policy")
    mean_rand, _ = run_policy(policy_random)
    print("optimal policy")
    mean_opt, _ = run_policy(policy_optimal)

    print(mean_eval, mean_rand, mean_opt)

    # print(policy_eval)
    # print(policy_random)
    # print(policy_optimal)

    # print(mean_opt, mean_rand, mean_eval)
    # import sys
    # sys.exit()

    # Normalize to [0, 1]
    if mean_opt != mean_rand:
        normalized = (mean_eval - mean_rand) / (mean_opt - mean_rand)
    else:
        normalized = 0.0  # avoid division by zero

    return normalized


import numpy as np

def evaluate_policy_normalized_parallel(policy_eval, policy_random, policy_optimal, P, r, S, A, gamma=0.9, num_episodes=1000, max_steps=100):
    normalized_returns = []

    for episode in range(num_episodes):
        s_eval = s_rand = s_opt = np.random.randint(S)
        returns = {'eval': 0.0, 'random': 0.0, 'optimal': 0.0}
        discounts = {'eval': 1.0, 'random': 1.0, 'optimal': 1.0}

        for step in range(max_steps):
            a_eval = np.random.choice(A, p=policy_eval[s_eval])
            a_rand = np.random.choice(A, p=policy_random[s_rand])
            a_opt  = np.random.choice(A, p=policy_optimal[s_opt])

            sprime_eval = np.random.choice(S, p=P[s_eval, a_eval] / P[s_eval, a_eval].sum())
            sprime_rand = np.random.choice(S, p=P[s_rand, a_rand] / P[s_rand, a_rand].sum())
            sprime_opt  = np.random.choice(S, p=P[s_opt, a_opt] / P[s_opt, a_opt].sum())

            r_eval = r[s_eval, a_eval, sprime_eval]
            r_rand = r[s_rand, a_rand, sprime_rand]
            r_opt  = r[s_opt, a_opt, sprime_opt]

            returns['eval'] += discounts['eval'] * r_eval
            returns['random'] += discounts['random'] * r_rand
            returns['optimal'] += discounts['optimal'] * r_opt

            s_eval, s_rand, s_opt = sprime_eval, sprime_rand, sprime_opt

        # Compute normalized return for this episode
        rand = returns['random']
        opt = returns['optimal']
        eval_ = returns['eval']

        if opt != rand:
            normalized =  (eval_ - rand) / (opt - rand)
        else:
            normalized = 0.0  # avoid divide-by-zero

        normalized_returns.append(normalized)

    mean_normalized = np.mean(normalized_returns)
    std_normalized = np.std(normalized_returns)

    print(f"Normalized return (per episode avg): {mean_normalized:.4f} ± {std_normalized:.4f}")
    return mean_normalized



def run_experiments(S, A, P, r,
                    gamma=0.9, alpha=0.01, beta=0.01,
                    risk_levels=[0.1, 0.5, 0.9],
                    data_sizes=[1000, 10000, 1000000],
                    seeds=[0, 1, 2]):
    results = []
    policy_random = np.ones((S, A)) / A  # uniform random policy

    for data_size in data_sizes:
        for tau in risk_levels:
            norm_beh, norm_box, norm_ch, norm_ell, norm_ell_9 = [], [], [], [], []

            for seed in seeds:
                np.random.seed(seed)
                print(f"Data Size: {data_size}, Tau: {tau}, Seed: {seed}")

                # Baseline optimal policy from expectile VI
                Qstar_vi, policy_vi = vi_expectile(tau, gamma, S, A, P, r)
                Qstar_opt, policy_opt = value_iteration(gamma, S, A, P, r, max_iter=10000, tol=1e-6)


                # Collect offline data from VI policy
                trajs = get_trajectories(0.2, S, A, P, r, policy_vi, data_size)
                print("Trajectories collected")

                # Train all policies
                # _, pol_naive = run_sacn(gamma, S, A, trajs, ensN=100)
                _, pol_box = run_sacn_box(gamma, alpha, S, A, trajs, ensN=100)
                _, pol_ch = run_sacn_convex_hull(gamma, alpha, S, A, trajs, ensN=100)
                _, pol_ell = run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs, quantile=1, ensN=100)
                _, pol_ell_9 = run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs, quantile=0.9, ensN=100)

                print("Policies trained")

                # Evaluate normalized returns [0, 1]
                norm_beh.append(evaluate_policy_normalized_parallel(policy_vi, policy_random, policy_opt, P, r, S, A, gamma))
                norm_box.append(evaluate_policy_normalized_parallel(pol_box, policy_random, policy_opt, P, r, S, A, gamma))
                norm_ch.append(evaluate_policy_normalized_parallel(pol_ch, policy_random, policy_opt, P, r, S, A, gamma))
                norm_ell.append(evaluate_policy_normalized_parallel(pol_ell, policy_random, policy_opt, P, r, S, A, gamma))
                norm_ell_9.append(evaluate_policy_normalized_parallel(pol_ell_9, policy_random, policy_opt, P, r, S, A, gamma))

                print("Evaluation done")

            # Aggregate results
            results.append({
                "tau": tau,
                "data_size": data_size,
                "box_norm": np.mean(norm_box),
                "box_std": np.std(norm_box),
                "ch_norm": np.mean(norm_ch),
                "ch_std": np.std(norm_ch),
                "ell_norm": np.mean(norm_ell),
                "ell_std": np.std(norm_ell),
                "ell_0.9_norm": np.mean(norm_ell_9),
                "ell_0.9_std": np.std(norm_ell_9),
                "beh_norm": np.mean(norm_beh),
                "beh_std": np.std(norm_beh),
            })

    # Save and return results
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rl_experiment_results_{timestamp}.csv"
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return df_results


import numpy as np
import matplotlib.pyplot as plt

def plot_entropy_vs_epoch(gamma, alpha, beta, S, A, trajs, seeds=[0, 1, 2],
                           ensN=100, batch_size=32, eta_q=0.1, eta_pi=0.1,
                           tau=0.005, max_iter=1000, tol=1e-6):
    def softmax(x, tau=1.0):
        x = x / tau
        x_max = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def compute_entropy(policy_logits):
        pi = softmax(policy_logits, 1.0)
        entropy = -np.sum(pi * np.log(pi + 1e-8), axis=1)
        return np.mean(entropy)

    algo_fns = {
        "Box": lambda trajs, seed: run_sacn_box(gamma, alpha, S, A, trajs, ensN=ensN, batch_size=batch_size,
                                                eta_q=eta_q, eta_pi=eta_pi, tau=tau, max_iter=max_iter, tol=tol),
        "Convex Hull": lambda trajs, seed: run_sacn_convex_hull(gamma, alpha, S, A, trajs, ensN=ensN, batch_size=batch_size,
                                                                eta_q=eta_q, eta_pi=eta_pi, tau=tau, max_iter=max_iter, tol=tol),
        "Ellipsoid": lambda trajs, seed: run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs, ensN=ensN, quantile=1.0,
                                                            batch_size=batch_size, eta_q=eta_q, eta_pi=eta_pi,
                                                            tau=tau, max_iter=max_iter, tol=tol),
        "Ellipsoid-0.9": lambda trajs, seed: run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs, ensN=ensN, quantile=0.9,
                                                                batch_size=batch_size, eta_q=eta_q, eta_pi=eta_pi,
                                                                tau=tau, max_iter=max_iter, tol=tol)
    }

    entropy_logs = {k: [] for k in algo_fns.keys()}

    for seed in seeds:
        np.random.seed(seed)
        print(f"Seed {seed}")

        for name, fn in algo_fns.items():
            print(f"  Running {name}...")
            Qs = np.random.rand(S, A, ensN) / (1 - gamma)
            Qs_target = np.copy(Qs)
            policy_logits = np.zeros((S, A))
            entropy_log = []

            for it in range(max_iter):
                batch_indices = np.random.choice(len(trajs), size=min(batch_size, len(trajs)), replace=False)
                batch = trajs[batch_indices]

                # Run Bellman backup + updates
                Qs_old = Qs.copy()
                for s, a, sprime, r in batch:
                    s, a, sprime = int(s), int(a), int(sprime)

                    if "Convex Hull" in name:
                        q_ensemble_next = Qs_target[sprime].T
                        pi_sprime = softmax(policy_logits[sprime][None, :])[0]
                        entropy = -np.sum(pi_sprime * np.log(pi_sprime + 1e-8))
                        values = np.dot(q_ensemble_next, pi_sprime)
                        i_star = np.argmin(values)
                        v_sprime = np.sum(q_ensemble_next[i_star] * pi_sprime)

                    elif "Ellipsoid" in name:
                        Q_samples = Qs_target[sprime]
                        q_mean = Q_samples.mean(axis=1)
                        q_centered = Q_samples - q_mean[:, None]
                        cov = np.dot(q_centered, q_centered.T) / (ensN - 1)
                        reg_cov = cov + 1e-6 * np.eye(A)
                        inv_cov = np.linalg.inv(reg_cov)
                        pi_sprime = softmax(policy_logits[sprime][None, :])[0]
                        entropy = -np.sum(pi_sprime * np.log(pi_sprime + 1e-8))
                        q_centered_T = q_centered.T
                        mahalanobis = np.einsum('bi,ij,bj->b', q_centered_T, inv_cov, q_centered_T)
                        q_val = 0.9 if "0.9" in name else 1.0
                        radius = np.sqrt(np.quantile(mahalanobis, q_val))
                        penalty = radius * np.linalg.norm(np.dot(cov, pi_sprime))
                        v_sprime = np.sum(pi_sprime * q_mean) - penalty

                    else:  # Box
                        q_next = np.min(Qs_target[sprime], axis=1)
                        pi_sprime = softmax(policy_logits[sprime][None, :])[0]
                        entropy = -np.sum(pi_sprime * np.log(pi_sprime + 1e-8))
                        v_sprime = np.sum(pi_sprime * q_next)

                    y = r + gamma * (v_sprime - alpha * entropy)
                    for i in range(ensN):
                        Qs[s, a, i] += eta_q * (y - Qs[s, a, i])

                    # Actor update
                    min_q_s = np.min(Qs[s], axis=1)
                    pi_s = softmax(policy_logits[s][None, :])[0]
                    grad_log_pi = pi_s * (min_q_s - np.sum(min_q_s * pi_s))
                    policy_logits[s] += eta_pi * (grad_log_pi - alpha * np.log(pi_s + 1e-8))

                Qs_target = tau * Qs + (1 - tau) * Qs_target
                entropy_log.append(compute_entropy(policy_logits))

                if np.linalg.norm(Qs - Qs_old) < tol:
                    break

            entropy_logs[name].append(entropy_log)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    for name, logs in entropy_logs.items():
        max_len = max(len(run) for run in logs)
        logs_padded = np.array([np.pad(run, (0, max_len - len(run)), mode='edge') for run in logs])
        mean_entropy = np.mean(logs_padded, axis=0)
        std_entropy = np.std(logs_padded, axis=0)

        plt.plot(mean_entropy, label=name)
        plt.fill_between(np.arange(max_len),
                         mean_entropy - std_entropy,
                         mean_entropy + std_entropy,
                         alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.title("Entropy vs Epoch (Mean ± Std over Seeds)")
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    environment_data_MR = [
        (1, 1, 1, 0.5, -5.0), (1, 1, 2, 0.4, 0.0), (1, 2, 1, 1.0, -100.0),
        (2, 1, 2, 0.5, -10.0), (2, 1, 3, 0.4, 0.0), (2, 1, 1, 0.1, 0.0), (2, 2, 1, 1.0, -100.0),
        (3, 1, 3, 0.5, -15.0), (3, 1, 4, 0.4, 0.0), (3, 1, 2, 0.1, 0.0), (3, 2, 1, 1.0, -100.0),
        (4, 1, 4, 0.5, -20.0), (4, 1, 5, 0.4, 0.0), (4, 1, 3, 0.1, 0.0), (4, 2, 1, 1.0, -100.0),
        (5, 1, 5, 0.5, -25.0), (5, 1, 6, 0.4, 0.0), (5, 1, 4, 0.1, 0.0), (5, 2, 1, 1.0, -100.0),
        (6, 1, 6, 0.5, -30.0), (6, 1, 7, 0.4, 0.0), (6, 1, 5, 0.1, 0.0), (6, 2, 1, 1.0, -100.0),
        (7, 1, 7, 0.5, -35.0), (7, 1, 8, 0.4, 0.0), (7, 1, 6, 0.1, 0.0), (7, 2, 1, 1.0, -100.0),
        (8, 1, 8, 0.5, -40.0), (8, 1, 9, 0.4, 0.0), (8, 1, 7, 0.1, 0.0), (8, 2, 1, 1.0, -100.0),
        (9, 1, 9, 0.5, -45.0), (9, 1, 10, 0.4, 0.0), (9, 1, 8, 0.1, 0.0), (9, 2, 1, 1.0, -100.0),
        (10, 1, 10, 0.9, -50.0), (10, 1, 9, 0.1, 0.0), (10, 2, 1, 1.0, -100.0)
    ]

    environment_data_RS = [
    (1, 1, 1, 1.0, 5.0), (1, 2, 1, 0.421657365594869, 0.0), (1, 2, 2, 0.578342634405131, 0.0),
    (2, 1, 1, 1.0, 5.0), (2, 2, 1, 0.137028976772708, 0.0), (2, 2, 2, 0.284628388822161, 0.0), (2, 2, 3, 0.578342634405131, 0.0),
    (3, 1, 2, 1.0, 5.0), (3, 2, 2, 0.137028976772708, 0.0), (3, 2, 3, 0.284628388822161, 0.0), (3, 2, 4, 0.578342634405131, 0.0),
    (4, 1, 3, 1.0, 5.0), (4, 2, 3, 0.137028976772708, 0.0), (4, 2, 4, 0.284628388822161, 0.0), (4, 2, 5, 0.578342634405131, 0.0),
    (5, 1, 4, 1.0, 5.0), (5, 2, 4, 0.137028976772708, 0.0), (5, 2, 5, 0.284628388822161, 0.0), (5, 2, 6, 0.578342634405131, 0.0),
    (6, 1, 5, 1.0, 5.0), (6, 2, 5, 0.137028976772708, 0.0), (6, 2, 6, 0.284628388822161, 0.0), (6, 2, 7, 0.578342634405131, 0.0),
    (7, 1, 6, 1.0, 5.0), (7, 2, 6, 0.137028976772708, 0.0), (7, 2, 7, 0.284628388822161, 0.0), (7, 2, 8, 0.578342634405131, 0.0),
    (8, 1, 7, 1.0, 5.0), (8, 2, 7, 0.137028976772708, 0.0), (8, 2, 8, 0.284628388822161, 0.0), (8, 2, 9, 0.578342634405131, 0.0),
    (9, 1, 8, 1.0, 5.0), (9, 2, 8, 0.137028976772708, 0.0), (9, 2, 9, 0.284628388822161, 0.0), (9, 2, 10, 0.578342634405131, 0.0),
    (10, 1, 9, 1.0, 5.0), (10, 2, 9, 0.137028976772708, 0.0), (10, 2, 10, 0.284628388822161, 0.0), (10, 2, 11, 0.578342634405131, 0.0),
    (11, 1, 10, 1.0, 5.0), (11, 2, 10, 0.137028976772708, 0.0), (11, 2, 11, 0.284628388822161, 0.0), (11, 2, 12, 0.578342634405131, 0.0),
    (12, 1, 11, 1.0, 5.0), (12, 2, 11, 0.137028976772708, 0.0), (12, 2, 12, 0.284628388822161, 0.0), (12, 2, 13, 0.578342634405131, 0.0),
    (13, 1, 12, 1.0, 5.0), (13, 2, 12, 0.137028976772708, 0.0), (13, 2, 13, 0.284628388822161, 0.0), (13, 2, 14, 0.578342634405131, 0.0),
    (14, 1, 13, 1.0, 5.0), (14, 2, 13, 0.137028976772708, 0.0), (14, 2, 14, 0.284628388822161, 0.0), (14, 2, 15, 0.578342634405131, 0.0),
    (15, 1, 14, 1.0, 5.0), (15, 2, 14, 0.137028976772708, 0.0), (15, 2, 15, 0.284628388822161, 0.0), (15, 2, 16, 0.578342634405131, 0.0),
    (16, 1, 15, 1.0, 5.0), (16, 2, 15, 0.137028976772708, 0.0), (16, 2, 16, 0.284628388822161, 0.0), (16, 2, 17, 0.578342634405131, 0.0),
    (17, 1, 16, 1.0, 5.0), (17, 2, 16, 0.137028976772708, 0.0), (17, 2, 17, 0.284628388822161, 0.0), (17, 2, 18, 0.578342634405131, 0.0),
    (18, 1, 17, 1.0, 5.0), (18, 2, 17, 0.137028976772708, 0.0), (18, 2, 18, 0.284628388822161, 0.0), (18, 2, 19, 0.578342634405131, 0.0),
    (19, 1, 18, 1.0, 5.0), (19, 2, 18, 0.137028976772708, 0.0), (19, 2, 19, 0.284628388822161, 0.0), (19, 2, 20, 0.578342634405131, 0.0),
    (20, 1, 19, 1.0, 5.0), (20, 2, 19, 0.137028976772708, 0.0), (20, 2, 20, 0.862971023227292, 86.2971023227292)
]


    S, A, P, r =  load_environment(environment_data_MR)

    # Q_vi, pi_vi = vi_expectile(0.1, 0.9, S, A, P, r)
    # trajs = get_trajectories(0.2, S, A, P, r, pi_vi, 10000)
    #
    # plot_entropy_vs_epoch(
    #     gamma=0.9,
    #     alpha=0.01,
    #     beta=0.01,
    #     S=S,
    #     A=A,
    #     trajs=trajs,
    #     seeds=[0, 1, 2,3,4,5],
    #     max_iter=50  # adjust for speed
    # )
    # import sys
    # sys.exit()

    policy_random = np.ones((S, A)) / A
    Qstar_optimal, policy_optimal = value_iteration(0.9, S, A, P, r, max_iter=100000, tol=1e-6)  #value_iteration(0.5, 0.9, S, A, P, r)
    Qstar_vi, policy_vi1 = vi_expectile(0.1, 0.9, S, A, P, r,max_iter=100000,tol=1e-6)
    Qstar_vi, policy_vi9 = vi_expectile(0.9, 0.9, S, A, P, r,max_iter=100000,tol=1e-6)

    print(policy_optimal)
    print(policy_vi1)
    print(policy_vi9)


    # print(evaluate_policy(policy_optimal, P, r, S, A, gamma=0.9, num_episodes=1000, max_steps=100))
    # print(evaluate_policy_normalized_parallel(policy_vi9, policy_random, policy_optimal, P, r, S, A,num_episodes=100, max_steps=100))

    df = run_experiments(
        S, A, P, r,
        gamma=0.9,
        alpha=0.01,
        beta=0.01,
        risk_levels=[0.1],  #,0.5, 0.9],  # or a subset like [0.2]
        data_sizes=[100] , #, 1000, 10000],  # depending on runtime
        seeds=[0]  #,1,2,3,4,5]  # for reproducibility
    )

    print(df)

    # gamma, alpha, beta = 0.9, 0.01, 0.01
    # risk_levels = [0.1, 0.5, 0.9]
    # data_sizes = [1000, 10000, 1000000]
    # seeds = [0, 1, 2]
    #
    # results = []
    #
    # for data_size in data_sizes:
    #     for tau in risk_levels:
    #         rewards_vi, rewards_box,rewards_ch, rewards_ell, rewards_ell_9 = [], [], [], [], []
    #
    #         for seed in seeds:
    #             np.random.seed(seed)
    #             print(data_size,tau,seed)
    #
    #             Qstar_vi, policy_vi = vi_expectile(tau, gamma, S, A, P, r)
    #
    #             trajs = get_trajectories(0.2, S, A, P, r, policy_vi, data_size)
    #             print("trajectories collected")
    #             # Train all policies
    #             Q_naive, pol_naive = run_sacn(gamma, S, A, trajs, ensN=100)
    #             Q_box, pol_box = run_sacn_box(gamma, alpha, S, A, trajs, ensN=100)
    #             Q_ch, pol_ch = run_sacn_convex_hull(gamma, alpha, S, A, trajs, ensN=100)
    #             Q_ell, pol_ell = run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs,quantile=1, ensN=100)
    #             Q_ell_9, pol_ell_9 = run_sacn_ellipsoid(gamma, alpha, beta, S, A, trajs, quantile=0.9, ensN=100)
    #
    #             print("collected policies")
    #             # Evaluate
    #             rewards_vi.append(evaluate_policy(policy_vi, P, r,S, A))
    #
    #             rewards_box.append(evaluate_policy(pol_box, P, r,S, A))
    #             rewards_ch.append(evaluate_policy(pol_ch, P, r,S, A))
    #             rewards_ell.append(evaluate_policy(pol_ell, P, r,S, A))
    #             rewards_ell_9.append(evaluate_policy(pol_ell_9, P, r,S, A))
    #             print("results done")
    #
    #         # Record mean/std
    #         results.append({
    #             "tau": tau,
    #             "data_size": data_size,
    #             "naive_mean": np.mean(rewards_vi),
    #             "naive_std": np.std(rewards_vi),
    #             "box_mean": np.mean(rewards_box),
    #             "box_std": np.std(rewards_box),
    #             "ch_mean": np.mean(rewards_ch),
    #             "ch_std": np.std(rewards_ch),
    #             "ellipsoid_mean": np.mean(rewards_ell),
    #             "ellipsoid_std": np.std(rewards_ell),
    #             "ellipsoid_0.9_mean": np.mean(rewards_ell_9),
    #             "ellipsoid_0.9_std": np.std(rewards_ell_9),
    #         })
    #
    # import pandas as pd
    # from datetime import datetime
    #
    # # Create DataFrame from results
    # df_results = pd.DataFrame(results)
    #
    # # Generate timestamp up to hhmmss
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #
    # # Save with timestamp
    # filename = f"rl_experiment_results_{timestamp}.csv"
    # df_results.to_csv(filename, index=False)
    # print(f"Results saved to {filename}")







    # Qstar_ra, policy_ra = vi_expectile(risk_tau, gamma, S, A, P, r)
    # trajs = get_trajectories(0.5, S, A, P, r, policy_ra, 1000000) #int(1e8))
    # # Get visitation frequencies
    # visitation_freq = get_state_action_visitation(trajs, S, A)
    #
    # # Print as table
    # print("State-Action Visitation Frequencies:")
    # for s in range(S):
    #     for a in range(A):
    #         print(f"State {s}, Action {a}: {visitation_freq[s, a]}")
    #
    # # Qstar_sac1, policy_sac1 = run_sacn(gamma, S, A, trajs, 1)
    # Qstar_sac10, policy_sac10 = run_sacn(gamma, S, A, trajs, 100)
    # # Qstar_sac100, policy_sac100 = run_sacn(gamma, S, A, trajs, 1000)
    #
    # Qstar_sacn100, policy_sacn100 = run_sacn_box(gamma, alpha, S, A, trajs, 100, batch_size=32,
    #              eta_q=0.1, eta_pi=0.1, tau=0.005, max_iter=5000, tol=1e-6)
    # print("Policy for VI Expectile:")
    # print(policy_ra)
    # print("Policy for SACN with 1 ensemble:")
    # # print(policy_sac1)
    #
    # print("Policy for SACN with 10 ensembles:")
    # print(policy_sac10)
    # print("Policy for SACN with 10 ensembles:")
    # # print(policy_sac100)
    # print(policy_sacn100)
    # print(Qstar_sacn100)

