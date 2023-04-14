import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Dict, MultiDiscrete

_ACTION_MASK = "action_mask"

class ActionNet(nn.Module):
    def __init__(self, 
                 observation_space : Dict, 
                 action_space : MultiDiscrete):
        super().__init__()
    
        self.observation_space = observation_space
        self.action_space = action_space
    
        self.layers = nn.Sequential(
            nn.Linear(self.get_flattened_obs_size(), 256),
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
        logits = self.layers(obs.reshape(1, -1))
        action_logits = [head(logits) for head in self.heads]

        concatenated_action_logits = torch.cat(action_logits, dim=-1)

        concatenated_action_logits_masked = torch.subtract(input = concatenated_action_logits, 
                                                           other = torch.tensor(1 - self.action_mask).reshape(1, -1), 
                                                           alpha = 1e7)
        
        self.values = self.value_head(logits)
        
        return torch.reshape(concatenated_action_logits_masked, [len(self.action_space), -1])

    # The functions below are from the original repo, sightly modified
    def get_flattened_obs_size(self) -> int:
        """Get the total size of the observation."""
        obs_size = 0
        for key in sorted(self.observation_space):
            if key == _ACTION_MASK:
                continue
            else:
                obs_size += np.prod(self.observation_space[key].shape)
        return int(obs_size)

    def get_flattened_obs(self, obs : Dict) -> torch.Tensor:
        """Get the flattened observation (ignore the action masks). """
        flattened_obs_dict = {}
        for key in sorted(self.observation_space):
            assert key in obs
            if key == _ACTION_MASK:
                self.action_mask = self.reshape_and_flatten_obs(obs[key])
            else:
                flattened_obs_dict[key] = self.reshape_and_flatten_obs(obs[key])
        
        #flattened_obs = torch.cat(list(flattened_obs_dict.values()), dim=-1) # this line is used when the observation is batchified
        #return torch.cat(list(flattened_obs_dict.values()), dim=-1)
        return torch.FloatTensor(list(flattened_obs_dict.values()))

    def reshape_and_flatten_obs(self, obs : torch.Tensor) -> torch.Tensor:
        """Flatten observation."""
        #print(type(obs))
        #assert len(obs.shape) >= 2
        batch_dim = obs.shape[0]
        return obs.reshape(batch_dim, -1)