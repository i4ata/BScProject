import torch
import torch.nn as nn

from typing import Optional, Tuple

class Critic(nn.Module):

    def __init__(self, state_space):
        super().__init__()

        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.fc1 = nn.Linear(state_space, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, 
                state: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        self.hidden_state = self.lstm(self.fc1(state), hidden_state if hidden_state else self.hidden_state) 
        return self.fc2(self.hidden_state[0])