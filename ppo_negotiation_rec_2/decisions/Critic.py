import torch
import torch.nn as nn

from typing import Tuple, Optional

class Critic(nn.Module):

    def __init__(self, state_space):
        super().__init__()

        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.input_layer = nn.Linear(state_space, 64)
        self.lstm = nn.LSTMCell(64, 64)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(64, 64)
            for i in range(3)
        ])
        
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, 
                state: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        self.hidden_state = self.lstm(self.input_layer(state), hidden_state if hidden_state else self.hidden_state) 
        x = self.hidden_state[0]
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output_layer(x)