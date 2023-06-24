import torch
import torch.nn as nn

from typing import Tuple, Optional

class Critic(nn.Module):

    def __init__(self, state_space, params: dict):
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