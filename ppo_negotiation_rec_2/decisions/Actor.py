import torch
import torch.nn as nn
from typing import Tuple, Optional

class Actor(nn.Module):

    def __init__(self, state_space, num_agents: int):
        super().__init__()

        self.hidden_state = None

        self.fc1 = nn.Linear(state_space, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.activation = nn.ReLU()

        self.actor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            for agent in range(num_agents - 1)
        ])

    def forward(self, 
                state: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        
        self.hidden_state = self.lstm(self.fc1(state), hidden_state if hidden_state else self.hidden_state)
        logits = self.activation(self.fc3(self.activation(self.fc2(self.hidden_state[0]))))
        return torch.stack([head(logits) for head in self.actor_heads], dim = 1)