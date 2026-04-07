import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor, EpistemicCritic
import numpy as np
import math
import copy


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
        hidden_size =  256
        self.num_critics = 1
        learning_rate = 5e-4*5
        self.clip_grad_param = 1

        self.z_dim = 3
        self.sigma0 = 1.0
        self.sigma = 0.1
        self.lam = 0.01

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
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # Replace with EpistemicCritic
        self.critic = EpistemicCritic(
            state_dim=state_size,  # state size
            action_dim=action_size,  # action size
            z_dim=self.z_dim,  # epistemic uncertainty dimension
            hidden_dim=hidden_size,  # hidden size for the critic network
            num_critics=self.num_critics,  # number of critics in the ensemble
        ).to(device)

        # Optimizer for the critic network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Create the target critic network (for target value learning)
        self.target_critic = EpistemicCritic(
            state_dim=state_size,  # state size
            action_dim=action_size,  # action size
            z_dim=self.z_dim,  # epistemic uncertainty dimension
            hidden_dim=hidden_size,  # hidden size for the critic network
            num_critics=self.num_critics,  # number of critics in the ensemble
        ).to(device)

        # Copy parameters from critic to target critic (for stable learning)
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

    def calc_policy_loss(self, states, done, beta):
        """
        Computes the loss:
        (1 / |B|) ∑_s [min_q ∑_a π_θ(a|s)q(a) - β∑_a π_θ(a|s) log π_θ(a|s)],
        where q*(a) = μ(s) - δ Σ(s)π_θ / sqrt(π_θ^T Σ(s) π_θ).

        Args:
            states: Tensor of states (batch_size, state_dim).
            done: Tensor indicating terminal states (batch_size,).
            beta: Entropy regularization coefficient.
            delta: Scaling parameter for the ellipsoid constraint.

        Returns:
            loss: Scalar representing the policy loss.
        """
        # Sample actions and policy probabilities
        action, action_probs, log_pis = self.actor_local.get_action(states)  # Shapes: (batch_size, num_actions)

        # Generate z for the ensemble critics
        z = torch.randn( states.size(0), self.z_dim, device=states.device)

        # Compute Q-values for the actions from the critics
        q_values, q_mean, q_cov = self.critic(states, action, z)  # Shape: (num_critics, batch_size, num_actions)

        # Compute the denominator for the ellipsoid penalty
        ellipsoid_denom = torch.einsum('bi,bij,bj->b', action_probs, q_cov, action_probs)  # Shape: (batch_size)
        ellipsoid_denom_sqrt = ellipsoid_denom.sqrt()  # Shape: (batch_size)

        delta = 1.0
        # Compute the penalty term for q*(a)
        penalty_term = torch.einsum('bij,bj->bi', q_cov, action_probs)  # Shape: (batch_size, num_actions)
        penalty_term = delta * penalty_term / ellipsoid_denom_sqrt.unsqueeze(-1)  # Shape: (batch_size, num_actions)

        # Compute q*(a)
        q_star = q_mean - penalty_term  # Shape: (batch_size, num_actions)

        # Compute the weighted sum ∑_a π_θ(a|s)q*(a)
        weighted_q_star = (1-done)*(action_probs * q_star).sum(dim=-1)  # Shape: (batch_size)

        # Compute the entropy term
        entropy_term = beta * (action_probs * log_pis).sum(dim=-1)  # Shape: (batch_size)

        # Combine terms for final loss
        loss = (-weighted_q_star + entropy_term).mean()

        batch_entropy = -log_pis.mean().item()
        q_value_std = q_values.std(0).mean().item()

        return loss, batch_entropy, q_value_std, log_pis

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
            # Generate a unique z for each critic
            z = torch.randn( state.size(0), self.z_dim,
                            device=state.device)  # [num_critics, batch_size, z_dim]


            # Sample context vectors c_i from unit sphere
            c_i = torch.randn(state.size(0), self.z_dim, device=state.device)  # [batch_size, z_dim]
            c_i = F.normalize(c_i, dim=-1)  # Normalize to unit sphere
            c_i = c_i.unsqueeze(0)#.repeat(self.num_critics, 1, 1)  # [num_critics, batch_size, z_dim]

            # Compute Gaussian bootstrapping term σc^Tz
            bootstrap_term = self.sigma0 * torch.sum(c_i * z, dim=-1, keepdim=True)  # [num_critics, batch_size, 1]

            q_next, q_mean, q_cov = self.target_critic(next_state, next_action,z)  #.min(0).values

            # Compute mean and covariance across ensemble dimension
            q_next_mean = q_mean  # Shape: (batch_size, num_actions)
            q_next_cov = q_cov

            # Compute the Mahalanobis ellipsoid penalty
            delta = 1  # Ellipsoid scaling factor
            weighted_probs = next_action_probs.unsqueeze(-1)  # Shape: (batch_size, num_actions, 1)
            covariance_term = torch.einsum('bai,bij,baj->b', weighted_probs, q_next_cov,
                                           weighted_probs)  # Shape: (batch_size)
            ellipsoid_penalty = delta * covariance_term.sqrt()  # Shape: (batch_size)

            # Ellipsoid-adjusted Q-value
            q_next_ellipsoid = ((q_next_mean - ellipsoid_penalty.unsqueeze(-1) ) * next_action_probs).sum(dim=-1)  # Shape: (batch_size)

            # Entropy regularization
            entropy = self.alpha.to(self.device) * torch.mul(next_action_probs, next_action_log_prob).sum(
                dim=1)  # Shape: (batch_size)

            # Q-target with ellipsoid constraint
            q_target = reward + self.gamma * (1 - done) * (q_next_ellipsoid - entropy).unsqueeze(1)  # Shape: (batch_size, 1)



        # z = torch.randn(self.num_critics, state.size(0), self.z_dim,
        #                     device=state.device)  # [num_critics, batch_size, z_dim]

            # Critic: Compute predicted Q-values
        q_values,_,_ = self.critic(state, action,z)  # Shape: (num_critics, batch_size, num_actions)

        # Ensure action is expanded along the batch dimension
        action_expanded = action#.unsqueeze(-1)  # Shape: (batch_size, 1)

        q_values_gathered = q_values.gather(1, action_expanded.long())  # Shape: (batch_size, 1)
        # q_values_gathered = q_values_gathered.squeeze(-1)  # Shape: (batch_size)


        # Compute loss using MSE
        self.qf_criterion = nn.MSELoss(reduction='none')

        loss = self.qf_criterion(q_values_gathered, q_target.detach()) #.unsqueeze(0))  # Shape: (num_critics, batch_size)
        loss = loss.mean(dim=(0)) #.sum()  # Mean over batch, sum over critics

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
            variance_loss =  -torch.mean(variance)

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
        # print(len(experiences))
        # import sys
        # sys.exit()

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

            q_random_std = 0 #self.critic(state, random_actions).std(0).mean().item()

        # # ---------------------------- update critic ---------------------------- #
        # # Get predicted next-state actions and Q values from target models
        # with torch.no_grad():
        #     _, action_probs, log_pis = self.actor_local.evaluate(next_states)
        #     Q_target1_next = self.critic1_target(next_states)
        #     Q_target2_next = self.critic2_target(next_states)
        #     Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)
        #
        #     # Compute Q targets for current states (y_i)
        #     Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1))

        # # Compute critic loss
        # q1 = self.critic1(states)
        # q2 = self.critic2(states)
        #
        # q1_ = q1.gather(1, actions.long())
        # q2_ = q2.gather(1, actions.long())
        #
        # critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        # critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)
        #
        # cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        # cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()

        # cql_alpha_loss = torch.FloatTensor([0.0])
        # cql_alpha = torch.FloatTensor([0.0])
        # if self.with_lagrange:
        #     cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
        #     cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
        #     cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)
        #
        #     self.cql_alpha_optimizer.zero_grad()
        #     cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
        #     cql_alpha_loss.backward(retain_graph=True)
        #     self.cql_alpha_optimizer.step()
        #
        # total_c1_loss = critic1_loss + cql1_scaled_loss
        # total_c2_loss = critic2_loss + cql2_scaled_loss

        # # Update critics
        # # critic 1
        # self.critic1_optimizer.zero_grad()
        # total_c1_loss.backward(retain_graph=True)
        # clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        # self.critic1_optimizer.step()
        # # critic 2
        # self.critic2_optimizer.zero_grad()
        # total_c2_loss.backward()
        # clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        # self.critic2_optimizer.step()
        #
        # # ----------------------- update target networks ----------------------- #
        # self.soft_update(self.critic1, self.critic1_target)
        # self.soft_update(self.critic2, self.critic2_target)

        return alpha_loss.item(), critic_loss.item(), actor_loss.item(), actor_batch_entropy, current_alpha, q_policy_std, q_random_std
