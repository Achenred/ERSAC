import torch
import torch.nn as nn
from torch.distributions import Categorical 
import numpy as np
import torch.nn.functional as F
import math


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs
    
    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)
        # dist = Categorical(action_probs)
        # action = dist.sample().to(state.device)
        action = torch.argmax(action_probs, dim=1)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)

        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_det_action(self, state):
        action_probs = self.forward(state)
        # dist = Categorical(action_probs)
        # action = dist.sample().to(state.device)
        action = torch.argmax(action_probs) #, dim=1)

        return action.detach().cpu()



class SetCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        super(SetCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Shared layers for feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU()
        )

        # Output heads: one for mean and one for covariance
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.covariance_head = nn.Linear(hidden_size, action_size * (action_size + 1) // 2)  # Predict lower-triangular elements

        # Initialize a learnable scale factor
        self.scale_factor = nn.Parameter(torch.tensor(1.))  # Learnable parameter

        # Parameter initialization

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases."""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0)

        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.covariance_head.weight, -3e-3, 3e-3)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> tuple:
        shared = self.shared_layers(x)
        mean = self.mean_head(shared)

        # Covariance: predict elements of the lower-triangular Cholesky decomposition
        cov_flat = self.covariance_head(shared)  # Shape: (batch_size, n_targets * (n_targets + 1) // 2)
        scale_factor = self.scale_factor
        # Build lower-triangular matrix
        L = torch.zeros(x.size(0), self.action_size, self.action_size, device=x.device)  # Ensure L is on the same device as x
        indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)
        L[:, indices[0], indices[1]] = cov_flat

        # Scale the covariance matrix using the learnable scale factor
        scaled_cov = (L @ L.transpose(1, 2))  # Reconstruct covariance and scale

        return mean,scale_factor, scaled_cov  # Return mean, Cholesky lower-triangular matrix, and scaled covariance




class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()

        self.critic = nn.Sequential(
            VectorizedLinear(state_dim , hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, action_dim, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [batch_size, state_dim + action_dim]

        state_action = state  #torch.cat([state, action.view(-1, 1)], dim=-1)
        # [num_critics, batch_size, state_dim + action_dim]
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values