import torch
import torch.nn as nn
from gym.spaces import MultiDiscrete

class Critic(nn.Module):

    def __init__(self, state_space: int, action_space: MultiDiscrete, params):
        super().__init__()

        self.input_layer = nn.Linear(state_space + action_space.nvec.sum(), params['hidden_size_critic'])
        self.layers = nn.ModuleList([
            nn.Linear(params['hidden_size_critic'], params['hidden_size_critic']) 
            for i in range(params['n_hidden_layers_critic'])])
        self.output_layer = nn.Linear(params['hidden_size_critic'], 1)
        self.activation = nn.ReLU()

    def forward(self, env_state: torch.Tensor, action_mask = torch.Tensor) -> torch.Tensor:

        x = self.input_layer(torch.cat((env_state, action_mask), dim = 1))
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)