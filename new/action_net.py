import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Dict, MultiDiscrete

_ACTION_MASK = "action_mask"
_FEATURES = "features"

class ActionNet(nn.Module):
    def __init__(self, 
                 observation_space : Dict, 
                 action_space : MultiDiscrete):
        super().__init__()
    
        self.observation_space = observation_space
        self.action_space = action_space
    
        self.layers = nn.Sequential(
            #nn.Linear(observation_space[_FEATURES].shape[0], 256),
            nn.Linear(observation_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.heads = nn.ModuleList([
            nn.Linear(256, space.n)
            for space in self.action_space
        ])

        self.value_head = nn.Linear(256, 1)

        self.action_mask = None
        self.values = None

    def forward(self, observation : Dict) -> torch.Tensor:
        obs = self.get_flattened_obs(observation)
        logits = self.layers(obs)
        action_logits = [head(logits) for head in self.heads]

        concatenated_action_logits = torch.cat(action_logits, dim=-1)

        concatenated_action_logits_masked = torch.subtract(input = concatenated_action_logits, 
                                                           other = torch.tensor(1 - self.action_mask).reshape(1, -1), 
                                                           alpha = 1e7)
        
        self.values = self.value_head(logits)
        
        return torch.reshape(concatenated_action_logits_masked, [len(self.action_space), -1])
    
    def get_flattened_obs(self, obs : Dict) -> torch.Tensor:
        """Get the flattened observation (ignore the action masks). """
        self.action_mask = obs[_ACTION_MASK]
        return torch.FloatTensor(obs[_FEATURES].reshape(1, -1))