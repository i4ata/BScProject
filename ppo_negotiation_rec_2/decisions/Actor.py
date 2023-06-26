import torch
import torch.nn as nn
from typing import Tuple, Optional

class Actor(nn.Module):

    def __init__(self, state_space: int, params: dict):
        super().__init__()

        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.lstm = nn.LSTMCell(params['hidden_size_actor'], params['hidden_size_actor'])
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor'])
            for i in range(params['n_hidden_layers_actor'])
        ])
        self.output_layer = nn.Linear(params['hidden_size_actor'], 1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

    def forward(self, 
                state: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        self.hidden_state = self.lstm(self.input_layer(state), hidden_state if hidden_state else self.hidden_state)
        x = self.hidden_state[0]
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.sigmoid(self.output_layer(x))