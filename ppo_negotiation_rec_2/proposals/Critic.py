import torch
import torch.nn as nn

from typing import Optional, Tuple

class Critic(nn.Module):

    def __init__(self, state_space, params: dict):
        super().__init__()

        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.input_layer = nn.Linear(state_space, params['hidden_size_critic'])
        self.lstm = nn.LSTMCell(params['hidden_size_critic'], params['hidden_size_critic'])
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_critic'], params['hidden_size_critic'])
            for i in range(params['n_hidden_layers_critic'])
        ])
        self.output_layer = nn.Linear(params['hidden_size_critic'], 1)
        self.activation = nn.Tanh()
    
    def forward(self, 
                state: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        self.hidden_state = self.lstm(self.input_layer(state), hidden_state if hidden_state else self.hidden_state) 
        x = self.hidden_state[0]
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        return self.output_layer(x)