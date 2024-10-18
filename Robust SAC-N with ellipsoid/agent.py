import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor, VectorizedCritic
import numpy as np
import math
import copy
import scipy.stats



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
        self.tau = 1e-2
        hidden_size = 256
        self.num_critics = 10
        learning_rate = 5e-4
        self.clip_grad_param = 1

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0

        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate*0.1)

        self.critic = VectorizedCritic(
            state_size, action_size, hidden_size, self.num_critics
        ).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.target_critic = VectorizedCritic(state_size, action_size, hidden_size, self.num_critics).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Critic Network (w/ Target Network)

        # self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        # self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        #
        # assert self.critic1.parameters() != self.critic2.parameters()
        #
        # self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        # self.critic1_target.load_state_dict(self.critic1.state_dict())
        #
        # self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        # self.critic2_target.load_state_dict(self.critic2.state_dict())
        #
        # self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        # self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        # self.softmax = nn.Softmax(dim=-1)

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, done, alpha):
        #TODO check if these values are correct accroding to the paper
        action, action_probs, log_pis = self.actor_local.get_action(states)

        q_value_dist = self.critic(states, action)

        assert q_value_dist.shape[0] == self.critic.num_critics

        # q_value_min = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -log_pis.mean().item()

        # temp = alpha.to(self.device) * log_pis
        # temp_unsqueezed = temp.unsqueeze(0)  # Shape becomes [1, 252, 2]
        # temp_replicated = temp_unsqueezed.expand(10, -1, -1)  # Shape becomes [10, 252, 2]
        # q_value_min = temp_replicated.min(0).values

        entropy = self.alpha.to(self.device) * torch.mul(action_probs, log_pis).sum(dim=1)

        action_probs_expanded = action_probs.unsqueeze(0).expand(10, -1, -1)  # Shape: [10, 256, 2]
        weighted_q_values = q_value_dist * action_probs_expanded  # Shape: [10, 256, 2]
        weighted_q_values_sum = weighted_q_values.sum(dim=2)  # Shape: [10, 256]
        q_values_min = weighted_q_values_sum.min(dim=0).values  # Shape: [256]
        loss = torch.mean((-q_values_min*(1-done) + entropy))

        return loss, batch_entropy, q_value_std, log_pis

    # def calc_policy_loss(self, states, done):
    #     #TODO check if these values are correct accroding to the paper
    #     action, action_probs, log_pis = self.actor_local.get_action(states)
    #
    #     q_value_dist = self.critic(states, action)
    #
    #     assert q_value_dist.shape[0] == self.critic.num_critics
    #
    #     # q_value_min = q_value_dist.min(0).values
    #     # needed for logging
    #     q_value_std = q_value_dist.std(0).mean().item()
    #     # Entropy of the policy (for logging purposes)
    #     entropy = self.alpha.to(self.device) * torch.mul(action_probs, log_pis).sum(dim=1)
    #
    #     # Expand action_probs to match the q_values shape [10, 256, 2]
    #     expanded_action_probs = action_probs.unsqueeze(0).expand(q_value_dist.shape[0], -1, -1)  # Shape: [10, 256, 2]
    #     weighted_q_values = expanded_action_probs * q_value_dist  # Shape: [10, 256, 2]
    #     expected_q_values = weighted_q_values.sum(dim=2)  # Shape: [10, 256]
    #     q_min = expected_q_values.min(dim=0)[0]  # Shape: [256]
    #
    #     q_target = -(1-done)*(q_min - entropy)
    #
    #     loss = torch.mean(q_target)
    #
    #     return loss, entropy, q_value_std, log_pis

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

            q_next = self.target_critic(next_state, next_action)  #.min(0).values


            # mean_q_values = torch.mean(q_next, dim=0)  # Shape: [batch_size, num_actions]
            # centered_q_values = q_next - mean_q_values  # Shape: [10, 256, 2]
            # # Covariance matrix for each state is (centered.T @ centered) / (num_ensembles - 1)
            # covariance_matrices = torch.einsum('ijk,ijl->jkl', centered_q_values, centered_q_values) / (
            #             q_next.size(0) - 1)
            # # Example: Set confidence level and degrees of freedom
            # confidence_level = 0.9
            # n_actions = q_next.shape[2]  # Number of actions
            # radius = torch.sqrt(torch.tensor(scipy.stats.chi2.ppf(confidence_level, df=n_actions)))
            #
            # # Reshape pi_theta to match the dimensions for matrix multiplication
            # pi_theta_expanded = next_action_probs.unsqueeze(-1)  # Shape: [batch_size, num_actions, 1]
            # quadratic_form = torch.matmul(torch.matmul(pi_theta_expanded.transpose(1, 2), covariance_matrices),
            #                               pi_theta_expanded)
            #
            # quadratic_form = quadratic_form.squeeze()
            #
            # q_min = reward.squeeze() + self.gamma*(torch.mul(next_action_probs, mean_q_values).sum(dim=1) + radius*torch.sqrt(quadratic_form))
            # entropy = self.alpha.to(self.device) * torch.mul(next_action_probs, next_action_log_prob).sum(dim=1)
            # q_target = q_min - entropy
            # Mean Q-values across the ensemble
            mean_q_values = torch.mean(q_next, dim=0)  # Shape: [batch_size, num_actions]

            # Center the Q-values by subtracting the mean
            centered_q_values = q_next - mean_q_values  # Shape: [10, 256, 2]

            # Compute covariance matrix using the centered Q-values
            covariance_matrices = torch.einsum('ijk,ijl->jkl', centered_q_values, centered_q_values) / (
                        q_next.size(0) - 1)

            # Compute inverse of covariance matrix
            inv_covariances = torch.inverse(covariance_matrices)

            # Flatten centered_q_values to prepare for Mahalanobis distance calculation
            centered_q_flat = centered_q_values.view(-1, centered_q_values.shape[-1])  # Shape: [2560, 2]
            inv_covariances_flat = inv_covariances.repeat(q_next.shape[0], 1, 1)  # Shape: [2560, 2, 2]

            # Compute Mahalanobis distance
            mahalanobis_distances = torch.sqrt(
                torch.einsum('bi,bij,bj->b', centered_q_flat, inv_covariances_flat, centered_q_flat))

            # confidence_level = 0.7
            # sorted_distances, _ = torch.sort(mahalanobis_distances)  # Sort distances in ascending order
            # index = int(confidence_level * len(sorted_distances)) -1  # Find the index corresponding to the confidence level
            # radius = sorted_distances[index]  # Get the radius at the 90th percentile
            # # print(radius)
            #
            # # Expand next_action_probs to prepare for matrix multiplication
            # pi_theta_expanded = next_action_probs.unsqueeze(-1)  # Shape: [batch_size, num_actions, 1]
            #
            # # Compute quadratic form: pi_theta^T @ covariance_matrix @ pi_theta
            # quadratic_form = torch.matmul(torch.matmul(pi_theta_expanded.transpose(1, 2), covariance_matrices),
            #                               pi_theta_expanded)
            # quadratic_form = quadratic_form.squeeze()  # Shape: [batch_size]
            #
            # # Compute worst-case Q-values (adding uncertainty adjustment)
            # q_min = reward.squeeze() + self.gamma * (1 - done) * (
            #         torch.mul(next_action_probs, mean_q_values).sum(dim=1) - radius * torch.sqrt(quadratic_form)
            # )
            #
            # # Calculate entropy for the policy
            # entropy = self.alpha.to(self.device) * torch.mul(next_action_probs, next_action_log_prob).sum(dim=1)
            #
            # # Final target Q-value, robust to uncertainty
            # q_target = q_min - entropy
            confidence_level = 0.7
            sorted_distances, _ = torch.sort(mahalanobis_distances, dim=0)  # Shape: [10, 256]
            index = int(confidence_level * q_next.size(0)) - 1  # Since we have 10 ensemble members
            radius_per_state = sorted_distances[index]  # Shape: [256], radius for each state at 70th percentile
            pi_theta_expanded = next_action_probs.unsqueeze(-1)  # Shape: [batch_size, num_actions, 1]
            quadratic_form = torch.matmul(torch.matmul(pi_theta_expanded.transpose(1, 2), covariance_matrices),
                                          pi_theta_expanded)
            quadratic_form = quadratic_form.squeeze()  # Shape: [batch_size]

            # Compute worst-case Q-values (adding uncertainty adjustment for each state)
            q_min = reward.squeeze() + self.gamma * (1 - done) * (
                    torch.mul(next_action_probs, mean_q_values).sum(dim=1) - radius_per_state * torch.sqrt(
                quadratic_form))

            entropy = self.alpha.to(self.device) * torch.mul(next_action_probs, next_action_log_prob).sum(dim=1)

            # Final target Q-value, robust to uncertainty
            q_target = q_min - entropy

        q_values = self.critic(state, action)

        # [ensemble_size, batch_size] - [1, batch_size]
        # q_values_ = q_values.gather(1, action.long())
        # Repeat action tensor for each head
        action_expanded = action.unsqueeze(0).expand(q_values.size(0), -1, -1)
        # Use gather with the expanded action tensor
        q_values_gathered = q_values.gather(2, action_expanded.long())

        # loss = (1/q_target.shape[0])*((q_values_gathered - q_target.view(1, -1)) ** 2).mean(dim=1).sum() #dim=0)
        self.qf_criterion = nn.MSELoss(reduction='none')
        loss = self.qf_criterion(q_values_gathered, q_target.detach().unsqueeze(0))

        loss = loss.mean(dim=(1, 2)).sum()

        self.eta = 0.0
        if self.eta > 0:
            # Step 1: Create an action mask
            action_expanded = action.unsqueeze(0).expand(q_values.size(0), -1, -1).long()  # [num_critics, batch_size, 1]
            action_mask = torch.ones_like(q_values, dtype=torch.bool)  # Create a mask of shape [num_critics, batch_size, num_actions]
            # Step 2: Mask out the selected actions
            action_mask.scatter_(2, action_expanded, False)  # Mark the selected actions as False (mask out)
            # Step 3: Gather Q-values for the remaining actions
            remaining_q_values = q_values.masked_select(action_mask).view(q_values.size(0), q_values.size(1), -1)  # [num_critics, batch_size, num_actions-1]
            mean_q_values = torch.mean(remaining_q_values, dim=0,keepdim=True)  # Shape: [1, batch_size, num_actions_remaining]
            variance = torch.mean((remaining_q_values - mean_q_values) ** 2,dim=0)  # Shape: [batch_size, num_actions_remaining]
            variance_loss =  torch.mean(variance)

            loss += self.eta * variance_loss
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
        actor_loss, actor_batch_entropy, q_policy_std, log_pis = self.calc_policy_loss(state, done, current_alpha)
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

            q_random_std = self.critic(state, random_actions).std(0).mean().item()


        return alpha_loss.item(), critic_loss.item(), actor_loss.item(), actor_batch_entropy, current_alpha, q_policy_std, q_random_std
