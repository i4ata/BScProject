import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self, state_space):
        super().__init__()

        self.hidden_state = None

        self.fc1 = nn.Linear(state_space, 64)
        self.lstm = nn.LSTMCell(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activation = nn.Tanh()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor: # state-value

        if self.hidden_state is None:
            z = torch.zeros(len(state), 64)
            self.hidden_state = (z, z)
        
        x = self.fc1(state)

        self.hidden_state = self.lstm(x, self.hidden_state)
        
        return self.fc3(self.activation(self.fc2(self.hidden_state[0])))