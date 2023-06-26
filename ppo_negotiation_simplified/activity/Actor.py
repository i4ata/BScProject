import torch
import torch.nn as nn
from gym.spaces import MultiDiscrete

from typing import List

class Actor(nn.Module):

    def __init__(self, state_space: int, action_space: MultiDiscrete, params: dict = None):
        super().__init__()

        # self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.input_layer = nn.Linear(state_space + action_space.nvec.sum(), params['hidden_size_actor'])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor']) 
            for i in range(params['n_hidden_layers_actor'])])
        self.activation = nn.Tanh()

        self.heads = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], space.n)
            for space in action_space
        ])

    def forward(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> List[torch.Tensor]:

        x = self.activation(self.input_layer(torch.cat((env_state, action_mask), dim = 1)))
        # x = self.activation(self.input_layer(env_state))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        action_mask = action_mask.reshape((env_state.size(0), len(self.heads), -1))

        action_logits = [torch.subtract(head(x), 1 - action_mask[:, i], alpha = 1e5)
                         for (i, head) in enumerate(self.heads)]
        return action_logits
        