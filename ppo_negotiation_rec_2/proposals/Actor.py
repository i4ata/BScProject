import torch
import torch.nn as nn
from typing import Tuple, Optional

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.hidden_state = None

        self.input_layer = nn.Linear(state_space, 64)
        self.lstm = nn.LSTMCell(64, 64)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(64, 64)
            for i in range(3)
        ])

        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

        self.output_layer = nn.ModuleDict({
            'proposal' : nn.Linear(64, action_space),
            'promise' : nn.Linear(64, action_space)
        })

    def forward(self, state : torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        self.hidden_state = self.lstm(self.input_layer(state), hidden_state if hidden_state else self.hidden_state)
        
        x = self.hidden_state[0]
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        proposal_probs = self.sigmoid(self.output_layer['proposal'](x))
        promise_probs = self.sigmoid(self.output_layer['promise'](x))

        proposal_probs = torch.cat(proposal_probs, dim = 1)
        promise_probs  = torch.cat(promise_probs,  dim = 1)

        return proposal_probs, promise_probs