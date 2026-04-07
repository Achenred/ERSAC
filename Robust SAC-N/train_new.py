import os
os.environ["WANDB_MODE"] = "disabled"

import torch.nn as nn
import pandas as pd
import gym
import wandb
from itertools import product
from scipy.stats import chi2
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import warnings
from collections import deque

# Suppress Gym deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress urllib3 OpenSSL warning
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_critics=5):
        super().__init__()
        self.num_critics = num_critics
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_critics)
        ])

    def forward(self, state):
        return torch.stack([critic(state) for critic in self.critics], dim=0)  # [N, B, A]


class EpinetCritic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_L_head = nn.Linear(hidden_dim, action_dim * latent_dim)

        self.sigma_P_head = nn.Linear(hidden_dim, action_dim * latent_dim)
        for param in self.sigma_P_head.parameters():
            param.requires_grad = False
        with torch.no_grad():
            for param in self.sigma_P_head.parameters():
                param.copy_(torch.randn_like(param) * 0.1)

    def forward(self, state):
        psi = self.feature_net(state)
        mu = self.mean_head(psi)
        return mu

    def get_mu_and_cov(self, state):
        B = state.size(0)
        psi = self.feature_net(state)
        mu = self.mean_head(psi)

        sigma_L = self.sigma_L_head(psi).view(B, self.action_dim, self.latent_dim)
        with torch.no_grad():
            sigma_P = self.sigma_P_head(psi).view(B, self.action_dim, self.latent_dim)
        sigma_total = sigma_L + sigma_P

        Sigma = torch.einsum('bad,bcd->bac', sigma_total, sigma_total)
        return mu, Sigma



import torch
import torch.nn as nn

class MuNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, psi_dim=64):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, psi_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(psi_dim, action_dim)

    def forward(self, state):
        psi = self.feature_net(state)     # [B, psi_dim]
        mu = self.mu_head(psi)            # [B, A]
        return mu, psi


class SigmaNetwork(nn.Module):
    def __init__(self, action_dim, latent_dim, psi_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.psi_dim = psi_dim
        self.action_dim = action_dim

        self.sigma_L_net = nn.Sequential(
            nn.Linear(psi_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.sigma_P_net = nn.Sequential(
            nn.Linear(psi_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        for param in self.sigma_P_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            for param in self.sigma_P_net.parameters():
                param.copy_(torch.randn_like(param) * 0.1)

    def forward(self, psi, z):
        """
        psi: [B, psi_dim]
        z: [N, B, latent_dim]
        Returns: [B, N, A]
        """
        N, B, d_z = z.shape

        psi_exp = psi.unsqueeze(0).expand(N, -1, -1)  # [N, B, psi_dim]
        input = torch.cat([psi_exp, z], dim=2)        # [N, B, psi + d_z]


        sigma_L = self.sigma_L_net(input)             # [N, B, A]
        with torch.no_grad():
            sigma_P = self.sigma_P_net(input)         # [N, B, A]
        sigma_total = sigma_L + sigma_P               # [N, B, A]
        return sigma_total.permute(1, 0, 2)           # [B, N, A]



def load_data(pkl_path):
    df = pd.read_pickle(pkl_path)
    states = torch.tensor(np.stack(df["state"].values), dtype=torch.float32)
    actions = torch.tensor(df["action"].values, dtype=torch.int64)
    rewards = torch.tensor(df["reward"].values, dtype=torch.float32)
    next_states = torch.tensor(np.stack(df["next_state"].values), dtype=torch.float32)
    dones = torch.tensor(df["done"].values.astype(float), dtype=torch.float32)
    return states, actions, rewards, next_states, dones

def evaluate_policy(actor, env, num_episodes=1):
    actor.eval()
    device = actor.device
    returns = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = actor(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        returns.append(total_reward)
    actor.train()
    return np.mean(returns)

def train_sac_n(actor, critics, target_critics, data, env, *,
                env_name,            # "CartPole-v1"
                model_type,          # "sacn_ch"
                data_size,           # len(dataset)
                tau,                 # the τ you sweep (string or float)
                wandb_log = True, gamma=0.9, alpha=0.2,
                actor_lr=1e-3, critic_lr=2e-3, batch_size=256, num_epochs=100000,
                up=0.05, eval_interval=50, save_dir = "trained_models"):
    os.makedirs(save_dir, exist_ok=True)
    best_windowed = -float("inf")
    eval_window = deque(maxlen=10)

    stale_iters = 0  # how many evals since last improvement

    M = 3000  # plateau patience (eval rounds)

    alpha_decay_step = 100
    alpha_decay_amount = 0.005

    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = actor.to(device)
    critics = critics.to(device)
    target_critics = target_critics.to(device)
    actor.device = device  # used in evaluation

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizers = [optim.Adam(critic.parameters(), lr=critic_lr) for critic in critics.critics]

    for epoch in range(num_epochs):
        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        indices = np.random.choice(len(states), batch_size)
        s = states[indices].to(device)
        a = actions[indices].to(device)
        r = rewards[indices].unsqueeze(1).to(device)
        s_ = next_states[indices].to(device)
        d = dones[indices].unsqueeze(1).to(device)

        with torch.no_grad():
            next_pi = actor(s_)
            # print(next_pi[0])
            next_qs = target_critics(s_)
            next_q = torch.min(next_qs, dim=0).values
            next_v = (next_pi * (next_q - alpha * torch.log(next_pi + 1e-8))).sum(dim=1, keepdim=True)
            target_q = (r + (1 - d) * gamma * next_v).detach()

        critic_losses = []
        for qnet, opt in zip(critics.critics, critic_optimizers):
            q_pred = qnet(s).gather(1, a.unsqueeze(1))
            loss_q = F.mse_loss(q_pred, target_q)
            opt.zero_grad();
            loss_q.backward();
            opt.step()
            critic_losses.append(loss_q.item())

        pi = actor(s)
        qs = critics(s)
        min_q = torch.min(qs, dim=0).values
        actor_loss = (pi * (alpha * torch.log(pi + 1e-8) - min_q)).sum(dim=1).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "critic_loss": np.mean(critic_losses),
                "actor_loss": actor_loss.item(),
                "entropy": -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean().item()
            })

        for t, s in zip(target_critics.critics, critics.critics):
            for tp, sp in zip(t.parameters(), s.parameters()):
                tp.data.copy_(up * sp.data + (1.0 - up) * tp.data)

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r = evaluate_policy(actor, env)
            print(f"[Epoch {epoch+1:04d}] "
                  f"Critic Loss: {np.mean(critic_losses):.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Eval Return: {avg_r:.2f} ")

            if wandb_log:
                wandb.log({ "eval_avg_return": avg_r })

            eval_window.append(avg_r)
            if len(eval_window) == eval_window.maxlen:
                window_mean = sum(eval_window) / eval_window.maxlen
                if window_mean > best_windowed:
                    best_windowed = window_mean
                    stale_iters = 0

                    actor_path = (f"{save_dir}/"
                                  f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt")
                    critic_path = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_critic.pt")

                    torch.save(actor.state_dict(), actor_path)
                    torch.save(critics.state_dict(), critic_path)
                    print(f"   ✓ saved new 10-eval best ({best_windowed:.1f})")

                    if wandb_log:
                        wandb.run.summary["best_10eval_avg_return"] = best_windowed

                else:
                    stale_iters +=1

                # plateau trigger
                if stale_iters >= M:
                    print(f"Early stopping: 10-eval average "
                          f"stagnant for {M} eval rounds "
                          f"(best = {best_windowed:.2f}).")
                    break

    return actor, critics

# --------------------------------------------------------------------
def train_sac_n_CH(
    actor, critics, target_critics, data, env, *,
    env_name,            # "CartPole-v1"
    model_type,          # "sacn_ch"
    data_size,           # len(dataset)
    tau,                 # the τ you sweep (string or float)
    save_dir="trained_models",
    wandb_log=True,
    gamma=0.9, alpha=0.2,
    actor_lr=1e-3, critic_lr=2e-3,
    batch_size=256, num_epochs=100_000,
    polyak=0.05, eval_interval=50):
    """
    SAC-N with convex-hull worst-case objective.
    Saves actor/critic weights whenever the running mean of the **latest
    10 evaluations** exceeds all previous 10-eval averages.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 10-eval running window
    eval_window   = deque(maxlen=10)
    best_windowed = -float("inf")

    # alpha decay schedule
    alpha_decay_step, alpha_decay_amount = 100, 0.005
    M = 3000
    stale_iters = 0

    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, critics, target_critics = actor.to(device), critics.to(device), target_critics.to(device)
    actor.device = device

    actor_opt   = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opts = [optim.Adam(c.parameters(), lr=critic_lr) for c in critics.critics]

    start = time.time()

    # ─────────────────── training loop ───────────────────────────
    for epoch in range(num_epochs):

        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size)
        s  = states[idx].to(device)
        a  = actions[idx].to(device)
        r  = rewards[idx].unsqueeze(1).to(device)
        s_ = next_states[idx].to(device)
        d  = dones[idx].unsqueeze(1).to(device)

        # -------- TD target (convex-hull worst-case) --------------
        with torch.no_grad():
            next_pi = actor(s_)                         # [B,A]
            next_qs = target_critics(s_)               # [N,B,A]
            convex_vals = torch.einsum('ba,nba->nb', next_pi, next_qs)   # [N,B]
            worst_case_q = torch.min(convex_vals, dim=0).values.unsqueeze(1)  # [B,1]
            target_q = r + (1 - d) * gamma * worst_case_q

        # -------- critic update ----------------------------------
        critic_losses = []
        for net, opt in zip(critics.critics, critic_opts):
            q_pred = net(s).gather(1, a.unsqueeze(1))           # [B,1]
            loss_q = F.mse_loss(q_pred, target_q)
            opt.zero_grad(); loss_q.backward(); opt.step()
            critic_losses.append(loss_q.item())

        # -------- actor update (robust) --------------------------
        pi = actor(s)                               # [B,A]
        qs = critics(s)                             # [N,B,A]
        convex_vals = torch.einsum('ba,nba->nb', pi, qs)        # [N,B]
        worst_case_q_pi = torch.min(convex_vals, dim=0).values  # [B]
        loss_pi = (alpha * (pi * torch.log(pi + 1e-8)).sum(1) - worst_case_q_pi).mean()

        actor_opt.zero_grad(); loss_pi.backward(); actor_opt.step()

        # -------- wandb -----------------------------------------
        if wandb_log:
            import wandb
            wandb.log({"epoch": epoch,
                       "critic_loss": np.mean(critic_losses),
                       "actor_loss":  loss_pi.item(),
                       "entropy": -(pi*torch.log(pi+1e-8)).sum(1).mean().item()})

        # -------- target Polyak update --------------------------
        with torch.no_grad():
            for tgt, src in zip(target_critics.critics, critics.critics):
                for tp, sp in zip(tgt.parameters(), src.parameters()):
                    tp.data.mul_(1.0 - polyak).add_(polyak * sp.data)

        # -------- evaluation & 10-eval checkpoint ---------------
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r = evaluate_policy(actor, env)
            print(f"[Ep {epoch+1:05d}]  Ret {avg_r:.1f}  "
                  f"Crit {np.mean(critic_losses):.3f}")

            if wandb_log:
                wandb.log({"eval_avg_return": avg_r})

            eval_window.append(avg_r)
            if len(eval_window) == eval_window.maxlen:
                window_mean = sum(eval_window) / eval_window.maxlen
                if window_mean > best_windowed:
                    best_windowed = window_mean

                    actor_path  = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt")
                    critic_path = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_critic.pt")

                    torch.save(actor.state_dict(),   actor_path)
                    torch.save(critics.state_dict(), critic_path)

                    print(f"   ✓  saved new 10-eval best ({best_windowed:.1f})")
                    if wandb_log:
                        wandb.run.summary["best_10eval_avg_return"] = best_windowed

                else:
                    stale_iters +=1

                # plateau trigger
                if stale_iters >= M:
                    print(f"Early stopping: 10-eval average "
                          f"stagnant for {M} eval rounds "
                          f"(best = {best_windowed:.2f}).")
                    break

    print(f"Finished in {time.time()-start:.1f}s | best 10-eval avg {best_windowed:.1f}")
    return actor, critics


def compute_worst_case_q_ellipsoid(q_values, policy_probs, quantile=1.0):
    """
    q_values: [N, A]  -   ensemble Q-values for one state
    policy_probs: [A]  -   π(a|s) for that state
    returns scalar worst-case value under ellipsoid set
    """
    assert q_values.dim() == 2 and policy_probs.dim() == 1
    N, A = q_values.shape

    mean_q = q_values.mean(dim=0)            # μ ∈ R^A
    diffs  = q_values - mean_q               # center
    cov_q  = (diffs.T @ diffs) / (N - 1)     # Σ
    cov_q += torch.eye(A, device=q_values.device) * 1e-4

    L = torch.linalg.cholesky(cov_q)         # Σ = L Lᵀ
    # Mahalanobis distance of each sample: ||L⁻¹(x-μ)||²
    solved  = torch.cholesky_solve(diffs.T, L)  #.unsqueeze(-1), L)   # [A,N,1]
    dists_sq = (diffs.T * solved.squeeze(-1)).sum(dim=0)       # [N]

    # radius ε from empirical quantile
    cutoff = int(np.ceil(quantile * N)) - 1
    epsilon = torch.sqrt(torch.kthvalue(dists_sq, cutoff + 1).values + 1e-8)

    expected_q = torch.dot(policy_probs, mean_q)               # π·μ
    # norm of π in Σ metric: ||L⁻¹ π||
    pi_sol  = torch.cholesky_solve(policy_probs.unsqueeze(1), L)
    pi_norm = torch.sqrt((policy_probs.unsqueeze(0) @ pi_sol).squeeze() + 1e-8)

    return expected_q - epsilon * pi_norm


# ─────────────────────── main training routine ────────────────────────
def train_sac_n_ellipsoid(
    actor, critics, target_critics, data, env, *,
    env_name,              # e.g. "CartPole-v1"
    model_type,            # e.g. "sacn_ellip"
    data_size,             # len(dataset)
    tau,                   # τ for filename
    save_dir="trained_models",
    wandb_log=True,
    gamma=0.9, alpha=0.2,
    actor_lr=1e-3, critic_lr=1e-3,
    batch_size=256, num_epochs=10_000,
    polyak=0.05, quantile=0.9, eval_interval=50):

    os.makedirs(save_dir, exist_ok=True)
    eval_window   = deque(maxlen=10)     # last 10 eval returns
    best_windowed = -float("inf")

    alpha_decay_step, alpha_decay_amount = 100, 0.005
    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor, critics, target_critics = actor.to(device), critics.to(device), target_critics.to(device)
    actor.device = device

    actor_opt   = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opts = [optim.Adam(c.parameters(), lr=critic_lr) for c in critics.critics]

    start = time.time()

    # ─────────────── training loop ────────────────
    for epoch in range(num_epochs):
        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size)
        s  = states[idx].to(device)
        a  = actions[idx].to(device)
        r  = rewards[idx].unsqueeze(1).to(device)
        s_ = next_states[idx].to(device)
        d  = dones[idx].unsqueeze(1).to(device)

        # ----- worst-case TD target -----
        with torch.no_grad():
            next_pi = actor(s_)                          # [B,A]
            next_qs = target_critics(s_)                 # [N,B,A]
            wc_next = []
            for b in range(s_.size(0)):
                wc_next.append(
                    compute_worst_case_q_ellipsoid(next_qs[:, b, :],
                                                  next_pi[b], quantile))
            wc_next = torch.stack(wc_next).unsqueeze(1)  # [B,1]
            entropy = -(next_pi * torch.log(next_pi + 1e-8)).sum(1, keepdim=True)
            target_q = r + (1 - d) * gamma * (wc_next - alpha * entropy)

        # ----- critic update -----
        critic_losses = []
        for net, opt in zip(critics.critics, critic_opts):
            q_pred = net(s).gather(1, a.unsqueeze(1))
            loss_q = F.mse_loss(q_pred, target_q)
            opt.zero_grad(); loss_q.backward(); opt.step()
            critic_losses.append(loss_q.item())

        # ----- actor update -----
        pi = actor(s)
        qs = critics(s)                                  # [N,B,A]
        loss_pi = 0
        for b in range(s.size(0)):
            q_vals = qs[:, b, :]
            pi_b   = pi[b]
            wc_val = compute_worst_case_q_ellipsoid(q_vals, pi_b, quantile)
            entropy_b = (pi_b * torch.log(pi_b + 1e-8)).sum()
            loss_pi += -(wc_val - alpha * entropy_b)
        loss_pi /= s.size(0)

        actor_opt.zero_grad(); loss_pi.backward(); actor_opt.step()

        if wandb_log:
            import wandb
            wandb.log({"epoch": epoch,
                       "critic_loss": np.mean(critic_losses),
                       "actor_loss":  loss_pi.item(),
                       "entropy": -(pi*torch.log(pi+1e-8)).sum(1).mean().item()})

        # target Polyak
        with torch.no_grad():
            for tgt, src in zip(target_critics.critics, critics.critics):
                for tp, sp in zip(tgt.parameters(), src.parameters()):
                    tp.data.mul_(1 - polyak).add_(polyak * sp.data)

        # ----- evaluation & checkpoint -----
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r = evaluate_policy(actor, env)
            print(f"[Ep {epoch+1:05d}] Ret {avg_r:.1f} "
                  f"Crit {np.mean(critic_losses):.3f}")

            if wandb_log:
                wandb.log({"eval_avg_return": avg_r})

            eval_window.append(avg_r)
            if len(eval_window) == eval_window.maxlen:
                window_mean = sum(eval_window) / eval_window.maxlen
                if window_mean > best_windowed:
                    best_windowed = window_mean

                    actor_path  = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt")
                    critic_path = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_critic.pt")

                    torch.save(actor.state_dict(),   actor_path)
                    torch.save(critics.state_dict(), critic_path)
                    print(f"   ✓ saved new 10-eval best ({best_windowed:.1f})")

                    if wandb_log:
                        wandb.run.summary["best_10eval_avg_return"] = best_windowed

    print(f"Finished in {time.time()-start:.1f}s | best 10-eval avg {best_windowed:.1f}")
    return actor, critics
'''

def batched_worst_case_q_ellipsoid(q_vals, pi, quantile=0.9, eps=1e-6):
    """
    Vectorised ellipsoid robust value for an entire batch.

    q_vals : [N, B, A]   ensemble Q-values for B states, N critics
    pi     : [B, A]      action probabilities for those B states
    Returns: [B]         worst-case robust value per state
    """
    N, B, A = q_vals.shape
    device  = q_vals.device

    # μ_b,a
    mean_q  = q_vals.mean(dim=0)                      # [B,A]
    diffs   = q_vals - mean_q.unsqueeze(0)            # [N,B,A]

    # Σ_b = (1/(N-1)) Σ_n (q_n - μ)(q_n - μ)ᵀ
    cov = torch.einsum('nba,nbc->bac', diffs, diffs) / (N - 1)   # [B,A,A]
    cov = cov + eps * torch.eye(A, device=device).unsqueeze(0)   # reg

    L = torch.linalg.cholesky(cov)                    # [B,A,A]
    pi_col  = pi.unsqueeze(-1)                        # [B,A,1]
    pi_sol  = torch.cholesky_solve(pi_col, L)         # Σ⁻¹π  via Cholesky
    pi_norm = torch.sqrt((pi_col.transpose(1, 2) @ pi_sol).squeeze(-1) + eps)  # [B]

    # Mahalanobis distance of each critic sample
    cov_inv  = torch.cholesky_inverse(L)              # [B,A,A]
    diffs_b  = diffs.permute(1,0,2)                  # [B,N,A]
    # dists_sq = torch.einsum('bna,bab,bnb->bn', diffs_b, cov_inv, diffs_b)  # [B,N]

    dists_sq = torch.einsum('bna,bac,bnc->bn', diffs_b, cov_inv, diffs_b)

    k = int(np.ceil(quantile * N)) - 1
    eps_b = torch.sqrt(torch.kthvalue(dists_sq, k+1, dim=1).values + eps)  # [B]

    exp_q = (pi * mean_q).sum(dim=1)                  # [B]
    return exp_q - eps_b * pi_norm                    # [B]


def train_sac_n_ellipsoid(
    actor, critics, target_critics, data, env, *,
    env_name, model_type, data_size, tau,
    save_dir="trained_models", wandb_log=True,
    gamma=0.9, alpha=0.2,
    actor_lr=1e-3, critic_lr=1e-3,
    batch_size=256, num_epochs=10_000,
    polyak=0.05, quantile=0.9, eval_interval=50):

    os.makedirs(save_dir, exist_ok=True)
    eval_window   = deque(maxlen=10)
    best_windowed = -float("inf")

    alpha_decay_step, alpha_decay_amount = 100, 0.005
    M = 300
    stale_iters = 0
    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor, critics, target_critics = actor.to(device), critics.to(device), target_critics.to(device)
    actor.device = device

    actor_opt   = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opts = [optim.Adam(c.parameters(), lr=critic_lr) for c in critics.critics]

    start = time.time()

    # ─────────────── training loop ────────────────
    for epoch in range(num_epochs):
        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size)
        s  = states[idx].to(device)
        a  = actions[idx].to(device)
        r  = rewards[idx].unsqueeze(1).to(device)
        s_ = next_states[idx].to(device)
        d  = dones[idx].unsqueeze(1).to(device)

        # ---------- worst-case TD target (batched) ----------
        with torch.no_grad():
            next_pi = actor(s_)                      # [B,A]
            next_qs = target_critics(s_)             # [N,B,A]
            wc_next = batched_worst_case_q_ellipsoid(
                          next_qs, next_pi, quantile)     # [B]
            wc_next = wc_next.unsqueeze(1)           # [B,1]
            entropy = -(next_pi * torch.log(next_pi + 1e-8)).sum(1, keepdim=True)
            target_q = r + (1 - d) * gamma * (wc_next - alpha * entropy)

        # ---------- critic update ----------
        critic_losses = []
        for net, opt in zip(critics.critics, critic_opts):
            q_pred = net(s).gather(1, a.unsqueeze(1))
            loss_q = F.mse_loss(q_pred, target_q)
            opt.zero_grad(); loss_q.backward(); opt.step()
            critic_losses.append(loss_q.item())

        # ---------- actor update ----------
        pi = actor(s)
        qs = critics(s)                              # [N,B,A]
        wc_vals = batched_worst_case_q_ellipsoid(qs, pi, quantile)  # [B]
        entropy_batch = (pi * torch.log(pi + 1e-8)).sum(1)
        loss_pi = -(wc_vals - alpha * entropy_batch).mean()

        actor_opt.zero_grad(); loss_pi.backward(); actor_opt.step()

        # ---------- wandb ----------
        if wandb_log:
            import wandb
            wandb.log({"epoch": epoch,
                       "critic_loss": np.mean(critic_losses),
                       "actor_loss":  loss_pi.item(),
                       "entropy": -entropy_batch.mean().item()})

        # ---------- target Polyak ----------
        with torch.no_grad():
            for tgt, src in zip(target_critics.critics, critics.critics):
                for tp, sp in zip(tgt.parameters(), src.parameters()):
                    tp.data.mul_(1 - polyak).add_(polyak * sp.data)

        # ---------- evaluation & 10-eval checkpoint ----------
        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r = evaluate_policy(actor, env)
            print(f"[Ep {epoch+1:05d}]  Ret {avg_r:.1f} "
                  f"Crit {np.mean(critic_losses):.3f}")

            if wandb_log:
                wandb.log({"eval_avg_return": avg_r})

            eval_window.append(avg_r)
            if len(eval_window) == eval_window.maxlen:
                window_mean = sum(eval_window) / eval_window.maxlen
                if window_mean > best_windowed:
                    best_windowed = window_mean

                    actor_path  = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt")
                    critic_path = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_critic.pt")

                    torch.save(actor.state_dict(),   actor_path)
                    torch.save(critics.state_dict(), critic_path)
                    print(f"   ✓ saved new 10-eval best ({best_windowed:.1f})")

                    if wandb_log:
                        wandb.run.summary["best_10eval_avg_return"] = best_windowed

                else:
                    stale_iters +=1

                # plateau trigger
                if stale_iters >= M:
                    print(f"Early stopping: 10-eval average "
                          f"stagnant for {M} eval rounds "
                          f"(best = {best_windowed:.2f}).")
                    break

    print(f"Done in {time.time()-start:.1f}s | best 10-eval avg {best_windowed:.1f}")
    return actor, critics

'''

'''
def initialize_c_vectors(states, latent_dim, seed=42):
    np.random.seed(seed)
    c_vecs = np.random.randn(len(states), latent_dim)
    c_vecs /= np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-8
    return torch.tensor(c_vecs, dtype=torch.float32)



def train_epinet_sac(actor, epinet, target_epinet, data, env, wandb_log = True, gamma=0.99, alpha=0.2,
                     actor_lr=1e-3, critic_lr=2e-3, batch_size=256,
                     num_epochs=10000, eval_interval=50, quantile=0.9, tau=0.05,
                     latent_dim=16, sigma_bar=0.1, n_samples=10):
    alpha_decay_step = 100
    alpha_decay_amount = 0.005

    states, actions, rewards, next_states, dones = data
    c_vectors_full = initialize_c_vectors(states, latent_dim)  # Sampled and fixed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, epinet, target_epinet = actor.to(device), epinet.to(device), target_epinet.to(device)
    actor.device = device

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr,  weight_decay=1e-4)

    epinet_optim = optim.Adam(filter(lambda p: p.requires_grad, epinet.parameters()), lr=critic_lr, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size)

        s = states[idx].to(device)
        a = actions[idx].to(device)
        r = rewards[idx].unsqueeze(1).to(device)
        s_ = next_states[idx].to(device)
        d = dones[idx].unsqueeze(1).to(device)
        c = c_vectors_full[idx].to(device)  # [B, latent_dim]

        z = torch.randn(n_samples, batch_size, latent_dim, device=device)  # [N, B, d_z]

        with torch.no_grad():
            next_pi = actor(s_)  # [B, A]
            mu_next, Sigma_next = target_epinet.get_mu_and_cov(s_)  # [B, A], [B, A, A]

            pi_col = next_pi.unsqueeze(2)

            maha = torch.sqrt(torch.bmm(torch.bmm(pi_col.transpose(1, 2), Sigma_next), pi_col).squeeze(-1) + 1e-8)
            chi_radius = np.sqrt(chi2.ppf(quantile, df=mu_next.size(1)))
            worst_case_value = (next_pi * mu_next).sum(1, keepdim=True) - chi_radius * maha
            entropy = -(next_pi * torch.log(next_pi + 1e-8)).sum(1, keepdim=True)
            next_v = worst_case_value - alpha * entropy
            target_q = r + (1 - d) * gamma * next_v


        mu, _ = epinet.get_mu_and_cov(s)
        psi = epinet.feature_net(s)
        sigma_L = epinet.sigma_L_head(psi).view(batch_size, epinet.action_dim, latent_dim)
        with torch.no_grad():
            sigma_P = epinet.sigma_P_head(psi).view(batch_size, epinet.action_dim, latent_dim)
        sigma = sigma_L + sigma_P


        z_exp = z.permute(1, 0, 2).unsqueeze(2)  # [B, N, 1, d_z]
        sigma_exp = sigma.unsqueeze(1)  # [B, 1, A, d_z]
        eps_q = torch.sum(sigma_exp * z_exp, dim=-1)  # [B, N, A]
        q_samples = mu.unsqueeze(1) + eps_q  # [B, N, A]

        a_selected = a.unsqueeze(1)  # [B, 1]
        q_sampled_selected = q_samples[torch.arange(batch_size), :, a_selected.squeeze()]  # [B, N]
        # z: [N, B, d_z], c: [B, d_z] → need to compute dot(z_n[b], c[b]) for each b and n
        c_exp = c.unsqueeze(0)  # [1, B, d_z]
        c_z = (z * c_exp).sum(-1).transpose(0, 1)  # [B, N]

        errors = q_sampled_selected - target_q - sigma_bar * c_z  # [B, N]
        critic_loss = (errors ** 2).mean()  # Scalar loss

        epinet_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(epinet.parameters(), max_norm=1.0)
        epinet_optim.step()

        pi = actor(s)  # [B, A]
        mu, Sigma = epinet.get_mu_and_cov(s)  # [B, A], [B, A, A]
        chi_radius = np.sqrt(chi2.ppf(quantile, df=pi.size(1)))


        pi_col = pi.unsqueeze(2)  # [B, A, 1]
        maha = torch.sqrt(torch.bmm(torch.bmm(pi_col.transpose(1, 2), Sigma), pi_col).squeeze(-1) + 1e-8)  # [B, 1]
        robust_q = (pi * mu).sum(dim=1, keepdim=True) - chi_radius * maha  # [B, 1]

        log_pi = torch.log(pi + 1e-8)
        entropy = (pi * log_pi).sum(dim=1, keepdim=True)  # [B, 1]

        actor_loss = (-robust_q + alpha * entropy).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optim.step()


        if wandb_log:
            wandb.log({
                "epoch": epoch,
                "critic_loss": np.mean(critic_loss.item()),
                "actor_loss": actor_loss.item(),
                "entropy": -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean().item()
            })

        for tp, sp in zip(target_epinet.parameters(), epinet.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

        if epoch % eval_interval == 0 or epoch ==0:
            avg_r, std_r = evaluate_policy(actor, env)
            print(f"[Epoch {epoch:05d}] Critic Loss: {critic_loss.item():.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | Return: {avg_r:.2f} ± {std_r:.2f}")

            if wandb_log:
                wandb.log({
                    "eval_avg_return": avg_r,
                    "eval_std_return": std_r
                })

    return actor, epinet
'''

def initialize_c_vectors(states, latent_dim, seed=42):
    np.random.seed(seed)
    c_vecs = np.random.randn(len(states), latent_dim)
    c_vecs /= np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-8
    return torch.tensor(c_vecs, dtype=torch.float32)

# ------------------------------------------------------------------
def train_epinet_sac(
    actor, epinet, target_epinet, data, env, *,
    env_name, model_type, data_size, tau,          # for filenames
    save_dir="trained_models", wandb_log=True,
    gamma=0.99, alpha=0.2,
    actor_lr=1e-3, critic_lr=2e-3,
    batch_size=256, num_epochs=10_000,
    eval_interval=50, quantile=0.9, polyak=0.05,
    latent_dim=16, sigma_bar=0.1, n_samples=10):

    os.makedirs(save_dir, exist_ok=True)
    eval_window   = deque(maxlen=10)
    best_windowed = -float("inf")

    alpha_decay_step, alpha_decay_amount = 100, 0.005
    M = 3000
    stale_iters = 0
    states, actions, rewards, next_states, dones = data
    c_vectors_full = initialize_c_vectors(states, latent_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, epinet, target_epinet = actor.to(device), epinet.to(device), target_epinet.to(device)
    actor.device = device

    actor_optim  = optim.Adam(actor.parameters(), lr=actor_lr,   weight_decay=1e-4)
    epinet_optim = optim.Adam(filter(lambda p: p.requires_grad, epinet.parameters()),
                              lr=critic_lr, weight_decay=1e-4)

    start = time.time()

    # ─────────────── training loop ────────────────
    for epoch in range(1, num_epochs + 1):

        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size, replace=False)
        s   = states[idx].to(device)
        a   = actions[idx].to(device)
        r   = rewards[idx].unsqueeze(1).to(device)
        s_  = next_states[idx].to(device)
        d   = dones[idx].unsqueeze(1).to(device)
        c   = c_vectors_full[idx].to(device)        # [B,d_z]

        z = torch.randn(n_samples, batch_size, latent_dim, device=device)

        # ---------- robust TD target ----------
        with torch.no_grad():
            next_pi = actor(s_)                                 # [B,A]
            mu_n, Sigma_n = target_epinet.get_mu_and_cov(s_)    # [B,A], [B,A,A]

            pi_col   = next_pi.unsqueeze(2)                     # [B,A,1]
            maha_n   = torch.sqrt( (pi_col.transpose(1,2) @ Sigma_n @ pi_col).squeeze(-1) + 1e-8 )
            chi_r    = np.sqrt(chi2.ppf(quantile, df=mu_n.size(1)))
            worst_q  = (next_pi * mu_n).sum(1, keepdim=True) - chi_r * maha_n
            entropy  = -(next_pi * torch.log(next_pi + 1e-8)).sum(1, keepdim=True)
            target_q = r + (1 - d) * gamma * (worst_q - alpha * entropy)

        # ---------- EPINET critic update ----------
        mu, _ = epinet.get_mu_and_cov(s)
        psi   = epinet.feature_net(s)
        sigma_L = epinet.sigma_L_head(psi).view(batch_size, epinet.action_dim, latent_dim)
        with torch.no_grad():
            sigma_P = epinet.sigma_P_head(psi).view(batch_size, epinet.action_dim, latent_dim)
        sigma = sigma_L + sigma_P                                   # [B,A,d_z]

        z_exp     = z.permute(1,0,2).unsqueeze(2)                   # [B,N,1,d_z]
        eps_q     = torch.sum(sigma.unsqueeze(1) * z_exp, dim=-1)   # [B,N,A]
        q_samples = mu.unsqueeze(1) + eps_q                         # [B,N,A]

        q_sampled_selected = q_samples[torch.arange(batch_size), :, a]   # [B,N]

        c_z   = (z * c.unsqueeze(0)).sum(-1).transpose(0,1)         # [B,N]
        errors = q_sampled_selected - target_q - sigma_bar * c_z
        critic_loss = (errors ** 2).mean()

        epinet_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(epinet.parameters(), 1.0)
        epinet_optim.step()

        # ---------- Actor update ----------
        pi    = actor(s)
        mu, Sigma = epinet.get_mu_and_cov(s)
        chi_r = np.sqrt(chi2.ppf(quantile, df=pi.size(1)))

        maha = torch.sqrt( (pi.unsqueeze(2).transpose(1,2) @ Sigma @ pi.unsqueeze(2)).squeeze(-1) + 1e-8 )
        robust_q = (pi * mu).sum(1, keepdim=True) - chi_r * maha
        entropy  = (pi * torch.log(pi + 1e-8)).sum(1, keepdim=True)

        actor_loss = (-robust_q + alpha * entropy).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optim.step()

        # ---------- wandb ----------
        if wandb_log:
            import wandb
            wandb.log({"epoch": epoch,
                       "critic_loss": critic_loss.item(),
                       "actor_loss":  actor_loss.item(),
                       "entropy": -entropy.mean().item()})

        # ---------- target Polyak ----------
        with torch.no_grad():
            for tp, sp in zip(target_epinet.parameters(), epinet.parameters()):
                tp.data.mul_(1 - polyak).add_(polyak * sp.data)

        # ---------- evaluation & checkpoint ----------
        if epoch % eval_interval == 0 or epoch == 1:
            avg_r = evaluate_policy(actor, env)
            print(f"[Ep {epoch:05d}] Crit {critic_loss.item():.3f} | "
                  f"Act {actor_loss.item():.3f} | Ret {avg_r:.1f}")

            if wandb_log:
                wandb.log({"eval_avg_return": avg_r})

            eval_window.append(avg_r)
            if len(eval_window) == eval_window.maxlen:
                window_mean = sum(eval_window) / eval_window.maxlen
                if window_mean > best_windowed:
                    best_windowed = window_mean

                    actor_path  = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt")
                    critic_path = (f"{save_dir}/"
                                   f"{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_epinet.pt")

                    torch.save(actor.state_dict(),  actor_path)
                    torch.save(epinet.state_dict(), critic_path)
                    print(f"   ✓ saved new 10-eval best ({best_windowed:.1f})")

                    if wandb_log:
                        wandb.run.summary["best_10eval_avg_return"] = best_windowed

                else:
                    stale_iters +=1

                # plateau trigger
                if stale_iters >= M:
                    print(f"Early stopping: 10-eval average "
                          f"stagnant for {M} eval rounds "
                          f"(best = {best_windowed:.2f}).")
                    break

    print(f"Done in {time.time()-start:.1f}s | best 10-eval avg {best_windowed:.1f}")
    return actor, epinet


'''

def compute_worst_case_q_ellipsoid(samples, pi, quantile=0.9):
    """
    samples: [N, A]
    pi: [A]
    Returns: worst-case value under Mahalanobis ellipsoid uncertainty
    """
    mean_q = samples.mean(dim=0)  # [A]
    cov_q = torch.cov(samples.T) + torch.eye(samples.size(1), device=samples.device) * 1e-4  # [A, A]
    cov_inv = torch.linalg.inv(cov_q)

    diffs = samples - mean_q  # [N, A]
    dists_sq = torch.einsum('ni,ij,nj->n', diffs, cov_inv, diffs)
    sorted_dists = torch.sort(dists_sq)[0]
    cutoff_index = int(np.ceil(quantile * samples.size(0))) - 1
    epsilon = torch.sqrt(sorted_dists[cutoff_index] + 1e-8)

    Sigma_pi = cov_q @ pi
    pi_cov_pi = pi @ Sigma_pi
    pi_norm = torch.sqrt(pi_cov_pi + 1e-8)

    robust_q = mean_q - epsilon * Sigma_pi / (pi_norm + 1e-8)
    return robust_q @ pi  # scalar


def train_epinet_sac_epinet_ellipsoid(actor, mu_net, sigma_net, target_mu_net, target_sigma_net,
                                      data, env, wandb_log=True, gamma=0.99, alpha=0.2,
                                      actor_lr=1e-3, critic_lr=2e-3, batch_size=256,
                                      num_epochs=10000, eval_interval=50, tau=0.05,
                                      quantile=0.9, n_samples=1000, latent_dim=16,
                                      alpha_decay_step=100, alpha_decay_amount=0.005,
                                      sigma_bar=0.1):
    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = actor.to(device)
    mu_net, sigma_net = mu_net.to(device), sigma_net.to(device)
    target_mu_net, target_sigma_net = target_mu_net.to(device), target_sigma_net.to(device)
    actor.device = device

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=1e-4)
    critic_optim = torch.optim.Adam(list(mu_net.parameters()) + list(sigma_net.parameters()),
                                    lr=critic_lr, weight_decay=1e-4)

    # Pre-sampled and fixed bootstrap directions for all states
    c_vectors_full = torch.randn(states.shape[0], latent_dim).to(device)
    c_vectors_full = F.normalize(c_vectors_full, dim=1)  # normalize if desired

    for epoch in range(num_epochs):
        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size)
        s = states[idx].to(device)
        a = actions[idx].to(device)
        r = rewards[idx].unsqueeze(1).to(device)
        s_ = next_states[idx].to(device)
        d = dones[idx].unsqueeze(1).to(device)
        c = c_vectors_full[idx].to(device)  # [B, latent_dim]

        z = torch.randn(n_samples, batch_size, latent_dim, device=device)

        with torch.no_grad():
            next_pi = actor(s_)  # [B, A]
            mu_next, psi_next = target_mu_net(s_)
            q_samples = target_sigma_net(psi_next, z) + mu_next.unsqueeze(1)  # [B, N, A]

            worst_case_q = []
            for b in range(s_.size(0)):
                q_vals = q_samples[b]  # [N, A]
                pi_b = next_pi[b]  # [A]
                wc = compute_worst_case_q_ellipsoid(q_vals, pi_b, quantile)
                worst_case_q.append(wc)

            worst_case_q = torch.stack(worst_case_q).unsqueeze(1)  # [B, 1]
            entropy = -(next_pi * torch.log(next_pi + 1e-8)).sum(dim=1, keepdim=True)
            target_q = r + (1 - d) * gamma * (worst_case_q - alpha * entropy)

        mu, psi = mu_net(s)
        q_samples = sigma_net(psi, z) + mu.unsqueeze(1)  # [B, N, A]
        a_selected = a.unsqueeze(1)
        q_sampled_selected = q_samples[torch.arange(batch_size), :, a_selected.squeeze()]  # [B, N]

        c_exp = c.unsqueeze(0)  # [1, B, d_z]
        c_z = (z * c_exp).sum(-1).transpose(0, 1)  # [B, N]

        errors = q_sampled_selected - target_q - sigma_bar * c_z  # [B, N]
        critic_loss = (errors ** 2).mean()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(mu_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), max_norm=1.0)
        critic_optim.step()

        pi = actor(s)  # [B, A]
        mu, psi = mu_net(s)
        q_samples = sigma_net(psi, z) + mu.unsqueeze(1)  # [B, N, A]
        actor_loss = 0

        for b in range(s.size(0)):
            q_vals = q_samples[b]  # [N, A]
            pi_b = pi[b]
            robust_q = compute_worst_case_q_ellipsoid(q_vals, pi_b, quantile)
            entropy = (pi_b * torch.log(pi_b + 1e-8)).sum()
            actor_loss += -(robust_q - alpha * entropy)

        actor_loss /= s.size(0)

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optim.step()

        if wandb_log:
            import wandb
            wandb.log({
                "epoch": epoch,
                "alpha": alpha,
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "entropy": -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean().item()
            })

        with torch.no_grad():
            for tp, sp in zip(target_mu_net.parameters(), mu_net.parameters()):
                tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
            for tp, sp in zip(target_sigma_net.parameters(), sigma_net.parameters()):
                tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r, std_r = evaluate_policy(actor, env)
            print(f"[Epoch {epoch+1:04d}] "
                  f"Critic Loss: {critic_loss.item():.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Eval Return: {avg_r:.2f} ± {std_r:.2f}")
            if wandb_log:
                wandb.log({
                    "eval_avg_return": avg_r,
                    "eval_std_return": std_r
                })

    return actor, (mu_net, sigma_net)

'''

def batched_worst_case_q_ellipsoid_epi(q_vals, pi, quantile=0.9, eps=1e-6):
    N, B, A = q_vals.shape
    device = q_vals.device

    mu   = q_vals.mean(dim=0)                       # [B,A]
    diff = q_vals - mu.unsqueeze(0)                 # [N,B,A]

    cov = torch.einsum('nba,nbc->bac', diff, diff) / (N-1)  # [B,A,A]
    cov += eps * torch.eye(A, device=device).unsqueeze(0)

    L = torch.linalg.cholesky(cov)                  # [B,A,A]
    pi_col = pi.unsqueeze(-1)                       # [B,A,1]
    pi_sol = torch.cholesky_solve(pi_col, L)
    pi_norm = torch.sqrt((pi_col.transpose(1,2) @ pi_sol).squeeze(-1) + eps)  # [B]

    cov_inv = torch.cholesky_inverse(L)
    diff_b  = diff.permute(1,0,2)                  # [B,N,A]
    dists_sq = torch.einsum('bna,bac,bnc->bn', diff_b, cov_inv, diff_b)

    k = int(np.ceil(quantile * N)) - 1
    eps_b = torch.sqrt(torch.kthvalue(dists_sq, k+1, dim=1).values + eps)

    exp_q = (pi * mu).sum(1)                       # [B]
    return exp_q - eps_b * pi_norm                 # [B]

def train_epinet_sac_epinet_ellipsoid(
    actor, mu_net, sigma_net, target_mu_net, target_sigma_net,
    data, env, *, env_name, model_type, data_size, tau,
    save_dir="trained_models", wandb_log=True,
    gamma=0.99, alpha=0.2,
    actor_lr=1e-3, critic_lr=2e-3,
    batch_size=256, num_epochs=10_000,
    eval_interval=50, polyak=0.05,
    quantile=0.9, n_samples=1000, latent_dim=16,
    alpha_decay_step=100, alpha_decay_amount=0.005,
    sigma_bar=0.1):

    # ─── bookkeeping ───────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    eval_window, best_window = deque(maxlen=10), -float("inf")
    stale_iters, M, EPS = 0, 300, 1e-3          # plateau params

    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor, mu_net, sigma_net = actor.to(device), mu_net.to(device), sigma_net.to(device)
    target_mu_net, target_sigma_net = target_mu_net.to(device), target_sigma_net.to(device)
    actor.device = device

    actor_opt  = optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=1e-4)
    critic_opt = optim.Adam(list(mu_net.parameters()) + list(sigma_net.parameters()),
                            lr=critic_lr, weight_decay=1e-4)

    c_vectors_full = F.normalize(torch.randn(len(states), latent_dim, device=device), dim=1)

    # ─── training loop ─────────────────────────────────────────
    for epoch in range(num_epochs):
        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        # minibatch indices
        idx = np.random.choice(len(states), batch_size, replace=False)
        s   = states[idx].to(device)
        a   = actions[idx].to(device)
        r   = rewards[idx].unsqueeze(1).to(device)
        s_  = next_states[idx].to(device)
        d   = dones[idx].unsqueeze(1).to(device)
        c   = c_vectors_full[idx]                       # [B,d_z]
        z   = torch.randn(n_samples, batch_size, latent_dim, device=device)

        # ---------- robust TD target ----------
        with torch.no_grad():
            next_pi = actor(s_)
            mu_n, psi_n = target_mu_net(s_)
            q_next = target_sigma_net(psi_n, z) + mu_n.unsqueeze(1)          # [B,N,A]
            wc_next = batched_worst_case_q_ellipsoid_epi(
                          q_next.permute(1,0,2), next_pi, quantile)          # [B]
            target_q = r + (1-d)*gamma*(wc_next.unsqueeze(1)
                       - alpha * (-(next_pi*next_pi.log()).sum(1,keepdim=True)))

        # ---------- critic update ----------
        mu, psi  = mu_net(s)
        q_samp   = sigma_net(psi, z) + mu.unsqueeze(1)                       # [B,N,A]
        # batch_ix = torch.arange(batch_size, device=device)
        # q_sel    = q_samp[batch_ix, :, a]
        batch_idx = torch.arange(batch_size, device=device)  # [B]
        q_sel = q_samp.gather(
            2,  # action dim
            a.view(-1, 1, 1).expand(-1, q_samp.size(1), 1)
        ).squeeze(-1)
        # [B,N]
        c_z      = (z * c.unsqueeze(0)).sum(-1).transpose(0,1)               # [B,N]
        critic_loss = ((q_sel - target_q - sigma_bar*c_z)**2).mean()

        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(mu_net.parameters(),    1.0)
        torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), 1.0)
        critic_opt.step()

        # ---------- actor update ----------
        pi       = actor(s)
        mu, psi  = mu_net(s)
        q_samp   = sigma_net(psi, z) + mu.unsqueeze(1)                       # [B,N,A]
        wc_vals  = batched_worst_case_q_ellipsoid_epi(
                       q_samp.permute(1,0,2), pi, quantile)                  # [B]
        actor_loss = (alpha*(pi* pi.log()).sum(1) - wc_vals).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()

        # ---------- polyak targets ----------
        with torch.no_grad():
            for tp, sp in zip(target_mu_net.parameters(), mu_net.parameters()):
                tp.data.mul_(1-polyak).add_(polyak*sp.data)
            for tp, sp in zip(target_sigma_net.parameters(), sigma_net.parameters()):
                tp.data.mul_(1-polyak).add_(polyak*sp.data)

        # ---------- evaluation / plateau / checkpoint ----------
        if (epoch+1) % eval_interval == 0 or epoch == 0:
            avg_r = evaluate_policy(actor, env)
            print(f"[Ep {epoch+1:05d}]  Ret {avg_r:.1f}  Crit {critic_loss.item():.3f}")

            eval_window.append(avg_r)
            if len(eval_window) == eval_window.maxlen:
                mean10 = sum(eval_window)/10
                if mean10 > best_window + EPS:          # improvement
                    best_window = mean10; stale_iters = 0
                    torch.save(actor.state_dict(),
                               f"{save_dir}/{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt")
                    torch.save({'mu': mu_net.state_dict(),
                                'sigma': sigma_net.state_dict()},
                               f"{save_dir}/{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_epinet.pt")
                    print(f"  ✓ new best 10-eval avg: {best_window:.1f}")
                else:
                    stale_iters += 1
                if stale_iters >= M:
                    print(f"Early stop: plateau {M} evals (best={best_window:.2f})")
                    break

    print(f"Finished; best 10-eval average {best_window:.1f}")
    return actor, (mu_net, sigma_net)


if __name__ == "__main__":


    os.makedirs("trained_models", exist_ok=True)

    # ---------------- experiment grid ----------------
    envs = [ "LunarLander-v2"]   #CartPole-v0
    models = [ "box", "convex_hull"] #, "ellipsoid" ,"box", "convex_hull"]  # "ellipsoid", "convex_hull","epinet_nl",
    data_sizes =   [100_000] #[1_000, 10_000, 100_000]
    tau_values = [0.1, 0.5, 0.9]
    seeds = [0]

    # -------------------------------------------------
    for env_name, model_type, data_size, tau, seed in product(
            envs, models, data_sizes, tau_values, seeds):

        print(f"\n=== {model_type} | size={data_size} | τ={tau} | seed={seed} ===")

        # ----- load dataset & env ------------------------------------------------
        pkl_file = f"data/{env_name}_samples_tau_{tau}_n_{data_size}.pkl"
        data = load_data(pkl_file)
        states, actions, _, _, _ = data
        state_dim = states.shape[1]

        env = gym.make(env_name)
        env.reset(seed=seed)
        action_dim = env.action_space.n

        # ----- W&B run -----------------------------------------------------------
        wandb_run = wandb.init(
            project="ERSAC",
            name=f"{env_name}_{model_type}_n{data_size}_τ{tau}_seed{seed}",
            config={"env": env_name, "model": model_type,
                    "tau": tau, "data_size": data_size, "seed": seed},
            reinit=True)

        # ----- create actor ------------------------------------------------------
        actor = Actor(state_dim, action_dim)

        # =============== model dispatch =========================================
        if model_type == "box":
            critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critic = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critic.load_state_dict(critics.state_dict())

            trained_actor, trained_critic = train_sac_n(
                actor, critics, target_critic,
                data=data, env=env,
                env_name=env_name, model_type=model_type, data_size=data_size, tau=tau,
                save_dir="trained_models",
                gamma=0.99, alpha=0.2,
                # actor_lr=1e-4, critic_lr=2e-4,
                actor_lr=1e-4, critic_lr=5e-4,
                batch_size=256, num_epochs=5000, up = 0.01,
                eval_interval=1, wandb_log=True)

        elif model_type == "convex_hull":
            critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critic = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critic.load_state_dict(critics.state_dict())

            trained_actor, trained_critic = train_sac_n_CH(
                actor, critics, target_critic,
                data=data, env=env,
                env_name=env_name, model_type=model_type, data_size=data_size, tau=tau,
                save_dir="trained_models",
                gamma=0.99, alpha=0.2,
                # actor_lr=1e-4, critic_lr=2e-4,
                actor_lr=1e-5, critic_lr=2e-5,
                batch_size=256, num_epochs=5000, polyak= 0.01, #0.005,
                eval_interval=1, wandb_log=True)

        elif model_type == "ellipsoid":
            critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critic = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critic.load_state_dict(critics.state_dict())

            trained_actor, trained_critic = train_sac_n_ellipsoid(
                actor, critics, target_critic,
                data=data, env=env,
                env_name=env_name, model_type=model_type, data_size=data_size, tau=tau,
                save_dir="trained_models",
                gamma=0.99, alpha=0.2,
                actor_lr=2e-4, critic_lr=2e-4,
                batch_size=256, num_epochs=5000,
                quantile=0.99, eval_interval=1, wandb_log=True)

        elif model_type == "epinet":
            epinet = EpinetCritic(state_dim, action_dim, latent_dim=4, hidden_dim=128)
            target_epinet = EpinetCritic(state_dim, action_dim, latent_dim=4, hidden_dim=128)

            trained_actor, trained_critic = train_epinet_sac(
                actor, epinet, target_epinet,
                data=data, env=env,
                env_name=env_name, model_type=model_type, data_size=data_size, tau=tau,
                save_dir="trained_models",
                gamma=0.99, alpha=0.2,
                actor_lr=1e-5, critic_lr=2e-5,
                batch_size=256, num_epochs=5000,
                latent_dim=32, sigma_bar=0.01, n_samples=100,
                eval_interval=1, quantile=0.9, wandb_log=True)

        elif model_type == "epinet_nl":
            psi_dim, latent_dim = 128, 4
            mu_net = MuNetwork(state_dim, action_dim, hidden_dim=128, psi_dim=psi_dim)
            sigma_net = SigmaNetwork(action_dim, latent_dim, psi_dim=psi_dim, hidden_dim=128)
            target_mu = MuNetwork(state_dim, action_dim, hidden_dim=128, psi_dim=psi_dim)
            target_sig = SigmaNetwork(action_dim, latent_dim, psi_dim=psi_dim, hidden_dim=128)

            trained_actor, trained_critic = train_epinet_sac_epinet_ellipsoid(
                actor, mu_net, sigma_net, target_mu, target_sig,
                data=data, env=env,
                env_name=env_name, model_type=model_type, data_size=data_size, tau=tau,
                save_dir="trained_models",
                gamma=0.99, alpha=0.2,
                actor_lr=1e-4, critic_lr=2e-4,
                batch_size=256, num_epochs=5_000,
                eval_interval=1, quantile=0.9,
                n_samples=100, latent_dim=latent_dim,
                wandb_log=True)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # ----- save to disk ------------------------------------------------------
        actor_path = f"last_stage_models/{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt"
        critic_path = f"last_stage_models/{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_critic.pt"

        torch.save(trained_actor.state_dict(), actor_path)

        # critic may be nn.Module OR (mu, sigma) tuple
        if isinstance(trained_critic, tuple):
            torch.save({'mu': trained_critic[0].state_dict(),
                        'sigma': trained_critic[1].state_dict()}, critic_path)
        else:
            torch.save(trained_critic.state_dict(), critic_path)

        # ----- tidy up -----------------------------------------------------------
        wandb_run.finish()
        env.close()







