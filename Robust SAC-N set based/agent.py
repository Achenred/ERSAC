import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import SetCritic, Actor #,Critic, VectorizedCritic
import numpy as np
import math
import copy
import scipy.stats
from torch.distributions.multivariate_normal import MultivariateNormal




class SACN(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 device
                 ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SACN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device

        self.gamma = 0.99
        self.tau = 1e-2*2
        hidden_size = 32  #256
        self.num_critics = 10
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate*0.1)

        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0

        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate*5)  #5)

        # Assuming `n_features` corresponds to the state size, and `n_targets` corresponds to the action size
        self.critic = SetCritic(state_size, action_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Initialize the target critic with the same structure
        self.target_critic = SetCritic(state_size, action_size).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        #TODO check if these values are correct accroding to the paper
        action, action_probs, log_pis = self.actor_local.get_action(states)

        Q_mean, scaling_factor, Q_scaled_cov = self.critic(states, action)

        # Reshape policy for matrix multiplication: [n_states, n_actions, 1]
        pi_reshaped = action_probs.unsqueeze(-1)
        # Compute the numerator: Σ(s) π(⋅|s)
        Sigma_pi = torch.matmul(Q_scaled_cov, pi_reshaped)  # Shape: [n_states, n_actions, 1]
        # Compute the quadratic form: π(⋅|s)^T Σ(s) π(⋅|s)
        pi_Sigma_pi = torch.matmul(pi_reshaped.transpose(-1, -2), Sigma_pi)  # Shape: [n_states, 1, 1]
        pi_Sigma_pi_sqrt = torch.sqrt(pi_Sigma_pi.squeeze(-1).squeeze(-1))  # Shape: [n_states]
        # Calculate the final expression for q*

        q_star = Q_mean*action_probs -  scaling_factor*(Sigma_pi.squeeze(-1) *action_probs / pi_Sigma_pi_sqrt.unsqueeze(-1))  # Shape: [n_states, n_actions]

        weighted_q_star =  (q_star).sum(dim=-1)  #(action_probs * q_star).sum(dim=-1)  # Shape: [n_states]
        entropy = alpha * (action_probs * log_pis).sum(dim=-1)  # Shape: [n_states]
        loss = (entropy - weighted_q_star).mean()  # Average over the batch (1/|B| sum)

        return loss, entropy, 0, log_pis

    def _critic_loss(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_state: torch.Tensor,
            done: torch.Tensor,
    ) -> torch.Tensor:
        with (torch.no_grad()):
            next_action, next_action_probs, next_action_log_prob = self.actor_local.get_action(next_state)

            Q_next_mean, scaling_factor, Q_next_scaled_cov = self.target_critic(next_state, next_action)  #.min(0).values
            pi_theta_expanded = next_action_probs.unsqueeze(-1)  # Shape: [batch_size, num_actions, 1]

            quad_term = pi_theta_expanded.transpose(1, 2) @ Q_next_scaled_cov @ pi_theta_expanded
            quad_term = torch.reciprocal(torch.sqrt(quad_term.squeeze(-1)))
            temp = (Q_next_scaled_cov @ pi_theta_expanded).squeeze(-1)
            cov_term =  temp * quad_term
            min_q = (Q_next_mean - scaling_factor*cov_term)*next_action_probs
            q_min_value = (min_q ).sum(dim = 1)

            entropy_term = -self.alpha.to(self.device) * (next_action_probs * torch.log(next_action_probs + 1e-8)).sum(
                dim=1)  # Shape: [batch_size]
            q_target = reward + self.gamma *(1-done)* q_min_value.unsqueeze(1) + entropy_term.unsqueeze(1)  # Shape: [batch_size]


        q_values_mean_original, scaling_factor, q_values_scaled_cov = self.critic(state, action)
        action_indices = action.long()

        q_values_mean = torch.gather(q_values_mean_original, 1, action_indices)

        batch_indices = torch.arange(q_values_mean.shape[0])
        selected_variances = q_values_scaled_cov[
            batch_indices, action_indices.squeeze(-1), action_indices.squeeze(-1)]
        selected_variances = selected_variances.unsqueeze(-1)
        uncertainties = torch.sqrt(selected_variances)  #.unsqueeze(1)
        q_values_min = (q_values_mean - uncertainties) *scaling_factor

        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(q_values_min, q_target)
        loss = loss.mean()

        self.eta = 0  #1e-9 #0.00001
        if self.eta > 0:
            # Step 1: Extract the diagonal elements (variances sigma_aa) for all observations
            sigma_diag = torch.diagonal(q_values_scaled_cov, dim1=1, dim2=2)  # Shape will be (k, n)
            # Step 2: Compute mu_a + sigma_aa and mu_a - sigma_aa for all observations at once
            mu_plus_sigma = q_values_mean_original + torch.sqrt(sigma_diag)*scaling_factor
            mu_minus_sigma = -(q_values_mean_original - torch.sqrt(sigma_diag)*scaling_factor)
            # Step 3: Concatenate the results along the last axis to form the shape (k, 2n)
            result = torch.cat((mu_plus_sigma, mu_minus_sigma), dim=1)

            values_at_indices = result.gather(1, action_indices)  # Shape (256, 1)
            # Step 2: Subtract the values from all elements in the corresponding rows
            diff = result - values_at_indices  # Broadcasting will happen here, resulting in shape (256, 4)
            # Step 3: Set the values at the specified indices to zero
            diff.scatter_(1, action_indices, 0)  # Set the subtracted values to zero
            diff_expanded = diff.unsqueeze(2)

            result_tensor = diff_expanded @ diff_expanded.transpose(2, 1)  # Shape (256, 4, 4)
            upper_triangular = torch.triu(result_tensor, diagonal=1)  # Exclude diagonal, keep upper triangle
            final_sum = upper_triangular.sum(dim=(1, 2)).unsqueeze(1) #*(1/(result.shape[1]-1))  # Shape (256, 1)
            EDAC_loss = torch.sum(final_sum)
            loss += self.eta * EDAC_loss
        # print(EDAC_loss)
        return loss

    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):

        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones = experiences
        state, action, reward, next_state, done = [arr.to(self.device) for arr in experiences]

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, actor_batch_entropy, q_policy_std, log_pis = self.calc_policy_loss(state, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            self.soft_update(self.target_critic, self.critic, self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = 1  #self.actor_local.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)

            q_random_std = 0  #self.critic(state, random_actions).std(0).mean().item()


        return alpha_loss.item(), critic_loss.item(), actor_loss.item(), actor_batch_entropy, current_alpha, q_policy_std, q_random_std
