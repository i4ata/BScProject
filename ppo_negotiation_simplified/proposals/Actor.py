import torch
import torch.nn as nn
from typing import Tuple

class Actor(nn.Module):
    def __init__(self, state_space, action_space, num_agents: int, params: dict):
        super().__init__()

        self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor'])
            for i in range(params['n_hidden_layers_actor'])
        ])
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.output_layer = nn.ModuleDict({
            'proposal' : nn.Linear(params['hidden_size_actor'], action_space),
            'promise' : nn.Linear(params['hidden_size_actor'], action_space)
        })

    def forward(self, state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.activation(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        proposal_probs = self.sigmoid(self.output_layer['proposal'](x))
        promise_probs = self.sigmoid(self.output_layer['promise'](x))

        return proposal_probs, promise_probs