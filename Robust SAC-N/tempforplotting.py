import os
os.environ["WANDB_MODE"] = "disabled"

import torch.nn as nn
import pandas as pd
import gym
import wandb
from itertools import product
from scipy.stats import chi2
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import warnings

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

def evaluate_policy(actor, env, num_episodes=5):
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
    return np.mean(returns), np.std(returns)

def train_sac_n(actor, critics, target_critics, data, env, wandb_log = True, gamma=0.9, alpha=0.2,
                actor_lr=1e-3, critic_lr=2e-3, batch_size=256, num_epochs=100000,
                tau=0.05, eval_interval=50):
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
        for i in range(critics.num_critics):
            q_pred = critics.critics[i](s)
            q_value = q_pred.gather(1, a.unsqueeze(1))
            critic_loss = F.mse_loss(q_value, target_q)
            critic_optimizers[i].zero_grad()
            critic_loss.backward()
            critic_optimizers[i].step()
            critic_losses.append(critic_loss.item())

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
                tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r, std_r = evaluate_policy(actor, env)
            print(f"[Epoch {epoch+1:04d}] "
                  f"Critic Loss: {np.mean(critic_losses):.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Eval Return: {avg_r:.2f} ± {std_r:.2f}")

            if wandb_log:
                wandb.log({
                    "eval_avg_return": avg_r,
                    "eval_std_return": std_r
                })

    return actor, critics

def train_sac_n_CH(
    actor, critics, target_critics, data, env, wandb_log = True,
    gamma=0.9, alpha=0.2,
    actor_lr=1e-3, critic_lr=2e-3, batch_size=256, num_epochs=100000,
    tau=0.05, eval_interval=50
):
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
            next_pi = actor(s_)  # [B, A]
            next_qs = target_critics(s_)  # [N, B, A]
            # Take convex combination over critics — worst-case per (s, a)
            convex_vals = torch.einsum('ba,nba->nb', next_pi, next_qs)  # [N, B]
            worst_case_q = torch.min(convex_vals, dim=0).values.unsqueeze(1)  # [B, 1]
            target_q = r + (1 - d) * gamma * worst_case_q  # [B, 1]

        critic_losses = []
        for i in range(critics.num_critics):
            q_pred = critics.critics[i](s)  # [B, A]
            q_value = q_pred.gather(1, a.unsqueeze(1))  # [B, 1]
            critic_loss = F.mse_loss(q_value, target_q)
            critic_optimizers[i].zero_grad()
            critic_loss.backward()
            critic_optimizers[i].step()
            critic_losses.append(critic_loss.item())

        pi = actor(s)  # [B, A]
        qs = critics(s)  # [N, B, A]
        # Robust policy improvement with worst-case convex Q-values
        convex_vals = torch.einsum('ba,nba->nb', pi, qs)  # [N, B]
        worst_case_q_pi = torch.min(convex_vals, dim=0).values  # [B]
        actor_loss = (alpha * (pi * torch.log(pi + 1e-8)).sum(dim=1) - worst_case_q_pi).mean()

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
                tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r, std_r = evaluate_policy(actor, env)
            print(f"[Epoch {epoch+1:04d}] "
                  f"Critic Loss: {np.mean(critic_losses):.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Eval Return: {avg_r:.2f} ± {std_r:.2f}")

            if wandb_log:
                wandb.log({
                    "eval_avg_return": avg_r,
                    "eval_std_return": std_r
                })

    return actor, critics



def compute_worst_case_q_ellipsoid(q_values, policy_probs, quantile=1.0):
    """
    q_values: [N, A]  - Q-value samples for a given state
    policy_probs: [A] - action probabilities at that state
    Returns: worst-case robust value (scalar)
    """

    assert q_values.dim() == 2 and policy_probs.dim() == 1
    N, A = q_values.shape

    # Empirical mean
    mean_q = q_values.mean(dim=0)  # [A]

    # Centered samples
    diffs = q_values - mean_q  # [N, A]

    # Fast covariance matrix computation using outer product trick
    cov_q = (diffs.T @ diffs) / (N - 1)  # [A, A]
    cov_q += torch.eye(A, device=q_values.device) * 1e-4  # regularization

    # Solve linear system instead of inverse: vᵗ Σ⁻¹ v = ||L⁻¹ v||² via Cholesky
    L = torch.linalg.cholesky(cov_q)  # Lower-triangular Cholesky factor
    diffs_sol = torch.cholesky_solve(diffs.T.unsqueeze(-1), L)  # [A, N, 1]
    dists_sq = (diffs.T * diffs_sol.squeeze(-1)).sum(dim=0)  # [N]

    # Quantile-based calibration
    cutoff_index = int(np.ceil(quantile * N)) - 1
    epsilon = torch.sqrt(torch.kthvalue(dists_sq, cutoff_index + 1).values + 1e-8)

    # Compute expected Q: ⟨π, μ⟩
    expected_q = torch.dot(policy_probs, mean_q)

    # Mahalanobis norm: ||L⁻¹ π||² = πᵗ Σ⁻¹ π
    pi_col = policy_probs.unsqueeze(1)  # [A, 1]
    pi_sol = torch.cholesky_solve(pi_col, L)  # [A, 1]
    pi_norm = torch.sqrt((pi_col.T @ pi_sol).squeeze() + 1e-8)

    return expected_q - epsilon * pi_norm



def train_sac_n_ellipsoid(actor, critics, target_critics, data, env, wandb_log = True,
                          gamma=0.9, alpha=0.2, actor_lr=1e-3, critic_lr=1e-3,
                          batch_size=256, num_epochs=10000, tau=0.05,
                          quantile=0.9, eval_interval=50):
    alpha_decay_step = 100
    alpha_decay_amount = 0.005

    states, actions, rewards, next_states, dones = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = actor.to(device)
    critics = critics.to(device)
    target_critics = target_critics.to(device)
    actor.device = device

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in critics.critics]

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
            next_pi = actor(s_)  # [B, A]
            next_qs = target_critics(s_)  # [N, B, A]
            worst_case_next_q = []
            for b in range(s_.size(0)):
                q_vals = next_qs[:, b, :]  # [N, A]
                pi_b = next_pi[b]  # [A]
                wc_q = compute_worst_case_q_ellipsoid(q_vals, pi_b, quantile=quantile)
                worst_case_next_q.append(wc_q)
            worst_case_next_q = torch.stack(worst_case_next_q).unsqueeze(1)  # [B, 1]
            entropy = -(next_pi * torch.log(next_pi + 1e-8)).sum(dim=1, keepdim=True)
            target_q = (r + (1 - d) * gamma * (worst_case_next_q - alpha * entropy))  #.detach()

        critic_losses = []
        for i in range(critics.num_critics):
            q_pred = critics.critics[i](s)  # [B, A]
            q_value = q_pred.gather(1, a.unsqueeze(1))  # [B, 1]
            loss = F.mse_loss(q_value, target_q)
            critic_optimizers[i].zero_grad()
            loss.backward()
            critic_optimizers[i].step()
            critic_losses.append(loss.item())

        pi = actor(s)  # [B, A]
        qs = critics(s)  # [N, B, A]
        actor_loss = 0
        for b in range(s.size(0)):
            q_vals = qs[:, b, :]  # [N, A]
            pi_b = pi[b]  # [A]

            mean_q = q_vals.mean(dim=0)  # [A]
            cov_q = torch.cov(q_vals.T) + torch.eye(pi_b.size(0), device=device) * 1e-4  # [A, A]
            cov_inv = torch.linalg.inv(cov_q)

            diffs = q_vals - mean_q
            dists_sq = torch.einsum('ni,ij,nj->n', diffs, cov_inv, diffs)
            sorted_dists = torch.sort(dists_sq)[0]
            cutoff_index = int(np.ceil(quantile * q_vals.size(0))) - 1
            epsilon = torch.sqrt(sorted_dists[cutoff_index] + 1e-8)
            # === End: Calibrated radius ===

            Sigma_pi = cov_q @ pi_b
            pi_cov_pi = pi_b @ Sigma_pi
            pi_norm = torch.sqrt(pi_cov_pi + 1e-8)

            # Use calibrated epsilon instead of chi_radius
            surrogate = mean_q - epsilon * Sigma_pi / (pi_norm + 1e-8)

            entropy = (pi_b * torch.log(pi_b.clamp(min=1e-8))).sum()
            actor_loss += -(surrogate @ pi_b - alpha * entropy)

        actor_loss = actor_loss / s.size(0)

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
                tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

        if (epoch + 1) % eval_interval == 0 or epoch == 0:
            avg_r, std_r = evaluate_policy(actor, env)
            print(f"[Epoch {epoch+1:04d}] "
                  f"Critic Loss: {np.mean(critic_losses):.4f} | "
                  f"Actor Loss: {actor_loss.item():.4f} | "
                  f"Eval Return: {avg_r:.2f} ± {std_r:.2f}")

            if wandb_log:
                wandb.log({
                    "eval_avg_return": avg_r,
                    "eval_std_return": std_r
                })

    return actor, critics


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


import time

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

    c_vectors_full = torch.randn(states.shape[0], latent_dim).to(device)
    c_vectors_full = F.normalize(c_vectors_full, dim=1)

    for epoch in range(num_epochs):
        start_total = time.time()

        if epoch % alpha_decay_step == 0:
            alpha = max(0.001, alpha - alpha_decay_amount)

        idx = np.random.choice(len(states), batch_size)
        s = states[idx].to(device)
        a = actions[idx].to(device)
        r = rewards[idx].unsqueeze(1).to(device)
        s_ = next_states[idx].to(device)
        d = dones[idx].unsqueeze(1).to(device)
        c = c_vectors_full[idx].to(device)

        z = torch.randn(n_samples, batch_size, latent_dim, device=device)

        start_target = time.time()
        with torch.no_grad():
            next_pi = actor(s_)
            mu_next, psi_next = target_mu_net(s_)
            q_samples = target_sigma_net(psi_next, z) + mu_next.unsqueeze(1)
            worst_case_q = [compute_worst_case_q_ellipsoid(q_samples[b], next_pi[b], quantile)
                            for b in range(s_.size(0))]
            worst_case_q = torch.stack(worst_case_q).unsqueeze(1)
            entropy = -(next_pi * torch.log(next_pi + 1e-8)).sum(dim=1, keepdim=True)
            target_q = r + (1 - d) * gamma * (worst_case_q - alpha * entropy)
        end_target = time.time()

        start_critic = time.time()
        mu, psi = mu_net(s)
        q_samples = sigma_net(psi, z) + mu.unsqueeze(1)
        a_selected = a.unsqueeze(1)
        q_sampled_selected = q_samples[torch.arange(batch_size), :, a_selected.squeeze()]
        c_exp = c.unsqueeze(0)
        c_z = (z * c_exp).sum(-1).transpose(0, 1)
        errors = q_sampled_selected - target_q - sigma_bar * c_z
        critic_loss = (errors ** 2).mean()

        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(mu_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(sigma_net.parameters(), max_norm=1.0)
        critic_optim.step()
        end_critic = time.time()

        start_actor = time.time()
        pi = actor(s)
        mu, psi = mu_net(s)
        q_samples = sigma_net(psi, z) + mu.unsqueeze(1)
        actor_loss = 0
        for b in range(s.size(0)):
            q_vals = q_samples[b]
            pi_b = pi[b]
            robust_q = compute_worst_case_q_ellipsoid(q_vals, pi_b, quantile)
            entropy = (pi_b * torch.log(pi_b + 1e-8)).sum()
            actor_loss += -(robust_q - alpha * entropy)
        actor_loss /= s.size(0)

        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optim.step()
        end_actor = time.time()

        if wandb_log:
            import wandb
            wandb.log({
                "epoch": epoch,
                "alpha": alpha,
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "entropy": -(pi * torch.log(pi + 1e-8)).sum(dim=1).mean().item(),
                "time/target": end_target - start_target,
                "time/critic": end_critic - start_critic,
                "time/actor": end_actor - start_actor,
                "time/total": time.time() - start_total
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
                  f"Eval Return: {avg_r:.2f} ± {std_r:.2f} | "
                  f"T_total: {time.time() - start_total:.2f}s | "
                  f"T_target: {end_target - start_target:.2f}s | "
                  f"T_critic: {end_critic - start_critic:.2f}s | "
                  f"T_actor: {end_actor - start_actor:.2f}s")

    return actor, (mu_net, sigma_net)





if __name__ == "__main__":


    # --- Configuration ---
    envs = ['LunarLander-v2']   #["CartPole-v0"]  #
    models = [ 'epinet_nl']  #, 'ellipsoid']   #'ellipsoid', 'box', 'convex_hull', 'epinet'
    data_sizes = [ 1000]  #, 100000]  #10000,
    tau_values = [0.5]  #[0.1, 0.9, 0.5]
    seeds = [0]  #,2,3]

    for env_name, model_type, data_size, tau,seed in product(envs,models,data_sizes, tau_values,seeds):
        print(f"\n=== Running model: {model_type}, data_size: {data_size}, tau: {tau}, seed: {seed} ===")

        # --- Load Data and Environment ---
        pkl_file = f"data/{env_name}_samples_tau_{tau}_n_{data_size}.pkl"
        data = load_data(pkl_file)
        states, actions, _, _, _ = data
        state_dim = states.shape[1]
        env = gym.make(env_name)
        action_dim = env.action_space.n

        wandb.init(project="epistemic-robust-sac", config={
            "env_name": env_name,
            "model_type": model_type,
            "tau": tau,
            "data_size": data_size,
            "seed": seed
        })

        # --- Model Instantiation ---
        actor = Actor(state_dim, action_dim)
        env.reset(seed=0)  # ensure env initialized before sampling actions

        if model_type == 'box':
            '''
            Lunar lander: actor_lr=1e-3, critic_lr=2e-3, tau = 0.05
                        epinet: actor_lr=1e-4 * 0.8, critic_lr=1e-4
            cartpole: actor_lr=1e-3, critic_lr=2e-3, tau = 0.01
            '''
            critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critics.load_state_dict(critics.state_dict())
            trained_actor, trained_critic = train_sac_n(actor, critics, target_critics, data, env, wandb_log = True, gamma=0.99, alpha=0.2,
                                                        actor_lr=1e-3, critic_lr=2e-3, batch_size=256, num_epochs=2000,
                                                        tau=0.01, eval_interval=50)

        elif model_type == 'convex_hull':
            critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critics.load_state_dict(critics.state_dict())
            trained_actor, trained_critic = train_sac_n_CH( actor, critics, target_critics, data, env, wandb_log = True,
                                                            gamma=0.99, alpha=0.2,
                                                            actor_lr=1e-3, critic_lr=2e-3, batch_size=256, num_epochs=2000,
                                                            tau=0.01, eval_interval=50)

        elif model_type == 'ellipsoid':
            critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critics = VectorizedCritic(state_dim, action_dim, num_critics=100)
            target_critics.load_state_dict(critics.state_dict())
            trained_actor, trained_critic = train_sac_n_ellipsoid(actor, critics, target_critics, data, env,
                                                                  gamma=0.99, alpha=0.2, actor_lr=1e-4, critic_lr=2e-3,
                                                                  batch_size=256, num_epochs=5000, tau=0.05,
                                                                  quantile=0.9, eval_interval=50, wandb_log=True)

        elif model_type == 'epinet':
            epinet = EpinetCritic(state_dim, action_dim)
            target_epinet = EpinetCritic(state_dim, action_dim)
            trained_actor, trained_critic = train_epinet_sac(actor, epinet, target_epinet, data, env,
                                                             gamma=0.99, alpha=0.2, actor_lr=1e-3*0.8 , critic_lr=3e-3,
                                                             batch_size=256, num_epochs=3000, eval_interval=50,
                                                             quantile=0.9, tau=0.01,
                                                             latent_dim=16, sigma_bar=0.001, n_samples=1000,
                                                             wandb_log=True)


        elif model_type == 'epinet_nl':
            psi_dim = 128
            latent_dim = 4
            mu_net = MuNetwork(state_dim, action_dim, hidden_dim=128, psi_dim= psi_dim)
            sigma_net = SigmaNetwork(action_dim, latent_dim= latent_dim, psi_dim=psi_dim, hidden_dim=128)

            # Target networks (for Polyak averaging)
            target_mu_net = MuNetwork(state_dim, action_dim, hidden_dim=128, psi_dim=psi_dim)
            target_sigma_net = SigmaNetwork(action_dim, latent_dim=latent_dim, psi_dim=psi_dim, hidden_dim=128)

            # Train using refactored 2-network Epinet SAC training loop
            trained_actor, trained_critic = train_epinet_sac_epinet_ellipsoid(
                actor, mu_net, sigma_net, target_mu_net, target_sigma_net,
                data, env,
                wandb_log=True, gamma=0.99, alpha=0.2,
                actor_lr=2e-3, critic_lr=2e-3, batch_size=256,
                num_epochs=10000, eval_interval=50, tau=0.02,
                quantile=0.9, n_samples=100, latent_dim=latent_dim
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # --- Save Trained Models ---
        actor_path = f"trained_models/{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_actor.pt"
        critic_path = f"trained_models/{env_name}_{model_type}_data_size_{data_size}_tau_{tau}_critic.pt"
        torch.save(trained_actor.state_dict(), actor_path)
        torch.save(trained_critic.state_dict(), critic_path)

        wandb.finish()






