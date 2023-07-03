import torch
import torch.nn as nn
import numpy as np

from typing import Tuple

class Actor(nn.Module):
    def __init__(self, state_space: int, action_space: int, params: dict):
        super().__init__()

        self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor'])
            for i in range(params['n_hidden_layers_actor'])
        ])
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.output_layer = nn.Linear(params['hidden_size_actor'], action_space)

    def forward(self, state : torch.Tensor) -> torch.Tensor:

        x = self.activation(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.sigmoid(self.output_layer(x))
    
class Critic(nn.Module):

    def __init__(self, state_space: int, params: dict):
        super().__init__()

        self.input_layer = nn.Linear(state_space, params['hidden_size_critic'])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_critic'], params['hidden_size_critic'])
            for i in range(params['n_hidden_layers_critic'])
        ])
        self.output_layer = nn.Linear(params['hidden_size_critic'], 1)
        self.activation = nn.Tanh()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        
        x = self.activation(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

class NegotiationNet(nn.Module):
    
    def __init__(self, state_space: int, action_space: int, params: dict):

        super(NegotiationNet, self).__init__()

        self.actor = Actor(state_space, action_space, params['actor'])
        self.critic = Critic(state_space, params['critic'])

    def act(self, env_state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():

            action_probs: torch.Tensor = self.actor(env_state)
            state_value: torch.Tensor = self.critic(env_state)
            actions = torch.bernoulli(action_probs)
            logprobs = torch.log(self.get_probs(action_probs, actions))

        return actions, logprobs, state_value

    def evaluate(self, 
                 env_state: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        action_probs: torch.Tensor = self.actor(env_state)
        state_value: torch.Tensor = self.critic(env_state)
        logprobs = torch.log(self.get_probs(action_probs, actions))
        entropies = self.get_entropy(action_probs)

        return logprobs, entropies, state_value

    def eval_act(self, env_state: torch.Tensor, deterministic = False) -> np.ndarray:
        with torch.no_grad():
            action_probs: torch.Tensor = self.actor(env_state)
            sample_fn = torch.round if deterministic else torch.bernoulli
        return sample_fn(action_probs).cpu().detach().numpy()

    def get_probs(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return p * x + (1 - p) * (1 - x)
    
    def get_entropy(self, p: torch.Tensor) -> torch.Tensor:
        return - p * torch.log2(p) - (1 - p) * torch.log2(1 - p)