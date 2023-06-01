import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from Interfaces import ActorCritic

import sys
sys.path.append("..")
from rice import Rice

from typing import Tuple, List

class ActivityNet(ActorCritic):
    def __init__(self, env : Rice, n_features: int, params : dict = None):
        
        super(ActivityNet, self).__init__()

        self.state_space = n_features
        self.action_space = env.action_space[0] #? fix that

        self.actor_layers = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.actor_heads = nn.ModuleList([
            nn.Linear(64, space.n)
            for space in self.action_space
        ])
        
        self.critic = nn.Sequential(
            nn.Linear(self.state_space + env.len_actions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act_deterministically(self, env_state : torch.Tensor, **kwargs) -> np.ndarray:
        with torch.no_grad():
            action_mask = kwargs['action_mask']
            action_logits = self._get_action_logits(env_state, action_mask)
        return torch.stack(list(map(torch.argmax, action_logits))).detach().cpu().numpy()
    
    def act_stochastically(self, env_state : torch.Tensor, **kwargs) -> np.ndarray:
        with torch.no_grad():
            action_mask = kwargs['action_mask']
            action_logits = self._get_action_logits(env_state, action_mask)
            distributions = [Categorical(logits = logits) for logits in action_logits]
        return torch.cat([d.sample() for d in distributions]).detach().cpu().numpy()

    def act(self, env_state : torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            
            action_mask = kwargs['action_mask']
            action_logits = self._get_action_logits(env_state, action_mask)
            
            distributions = [Categorical(logits = logits) for logits in action_logits]
            actions = torch.stack([dist.sample().detach() for dist in distributions])
            actions_logprobs = torch.stack([dist.log_prob(action).detach() for (dist, action) in zip(distributions, actions)])
            state_values = self.critic(torch.cat((env_state, action_mask), dim = 1)).detach()

        return actions.T, actions_logprobs.T, state_values

    def evaluate(self, 
                 env_state: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        action_mask = kwargs['action_mask']
        actions = kwargs['actions']

        action_logits = self._get_action_logits(env_state, action_mask)
        
        distributions = [Categorical(logits = logits) for logits in action_logits]
        
        actions_logprobs = torch.stack([dist.log_prob(action) for (dist, action) in zip(distributions, actions.T)]).T
        distribution_entropies = torch.stack([dist.entropy() for dist in distributions]).T
        state_values = self.critic(torch.cat((env_state, action_mask), dim = 1))
        
        return actions_logprobs, state_values, distribution_entropies

    def _get_action_logits(self, env_state : torch.Tensor, action_mask: torch.Tensor) -> List[torch.Tensor]:

        action_mask = action_mask.reshape((env_state.shape[0], len(self.actor_heads), -1))

        logits = self.actor_layers(env_state)
        action_logits = [torch.subtract(head(logits), 1 - action_mask[:, i], alpha = 1e5)
                         for (i, head) in enumerate(self.actor_heads)]
        return action_logits

    def get_actor_parameters(self):
        return list(self.actor_layers.parameters()) + list(self.actor_heads.parameters())