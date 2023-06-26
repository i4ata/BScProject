import torch
import torch.nn as nn
from typing import Tuple, Optional

class Actor(nn.Module):
    def __init__(self, state_space: int, action_space: int, params: dict):
        super().__init__()

        self.hidden_state = None

        self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.lstm = nn.LSTMCell(params['hidden_size_actor'], params['hidden_size_actor'])
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor'])
            for i in range(params['n_hidden_layers_actor'])
        ])

        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

        self.output_layer = nn.ModuleDict({
            'proposal' : nn.Linear(params['hidden_size_actor'], action_space),
            'promise' : nn.Linear(params['hidden_size_actor'], action_space)
        })

    def forward(self, state : torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        self.hidden_state = self.lstm(self.input_layer(state), hidden_state if hidden_state else self.hidden_state)
        
        x = self.hidden_state[0]
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        proposal_probs = self.sigmoid(self.output_layer['proposal'](x))
        promise_probs = self.sigmoid(self.output_layer['promise'](x))

        return proposal_probs, promise_probs