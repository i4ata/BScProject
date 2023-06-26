import torch
import torch.nn as nn
from typing import Tuple, Optional

class Actor(nn.Module):

    def __init__(self, state_space: int, params: dict):
        super().__init__()

        self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor'])
            for i in range(params['n_hidden_layers_actor'])
        ])
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.output_layer = nn.Linear(params['hidden_size_actor'], 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        
        x = self.activation(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.sigmoid(self.output_layer(x))