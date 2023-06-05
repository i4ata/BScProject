import torch
import torch.nn as nn
from typing import Tuple

class Actor(nn.Module):
    def __init__(self, state_space, action_space, num_agents):
        super().__init__()

        self.hidden_state = None
    
        self.fc1 = nn.Linear(in_features=state_space, out_features=64)
        self.lstm = nn.LSTMCell(64, 64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.activation = nn.Tanh()

        self.actor_heads = nn.ModuleList([
            nn.ModuleDict({
                'decision' : nn.Sequential(
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                ),
                'proposal' : nn.Sequential(
                    nn.Linear(64, action_space),
                    nn.Sigmoid()
                ),
                'promise' : nn.Sequential(
                    nn.Linear(64, action_space),
                    nn.Sigmoid()
                )
            })
            for agent in range(num_agents - 1)
        ])

    def forward(self, state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.hidden_state is None:
            z = torch.zeros(len(state), 64)
            self.hidden_state = (z, z)

        x = self.fc1(state)

        self.hidden_state = self.lstm(x, self.hidden_state)
        logits = self.activation(self.fc3(self.activation(self.fc2(self.hidden_state[0]))))

        decision_probs, proposal_probs, promise_probs = zip(
            *[(head['decision'](logits), head['proposal'](logits), head['promise'](logits)) 
            for head in self.actor_heads]
        )

        decision_probs = torch.stack(decision_probs, dim = 1)
        proposal_probs = torch.stack(proposal_probs, dim = 1)
        promise_probs =  torch.stack(promise_probs, dim = 1)

        return decision_probs, proposal_probs, promise_probs