import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from PPO_negotiation import ActorCritic

import sys
sys.path.append("..")
from rice import Rice

from typing import Tuple

class ActivityNet(ActorCritic):
    def __init__(self, env : Rice, n_features: int, params : dict = None, device : str = None):
        
        super(ActivityNet, self).__init__()

        self.state_space = n_features
        self.action_space = env.action_space
        self.action_mask = env.default_agent_action_mask

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
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
    
    def act_deterministically(self, state : torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.actor_layers(state)
            action_logits = [head(logits) for head in self.actor_heads]
        return torch.stack(list(map(torch.argmax, action_logits))).detach().cpu().numpy()
    
    def act_stochastically(self, state : torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.actor_layers(state)
            action_logits = [head(logits) for head in self.actor_heads]
            distributions = [Categorical(logits = logits) for logits in action_logits]
        return torch.cat([d.sample() for d in distributions]).detach().cpu().numpy()

    def act(self, state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.actor_layers(state)
            action_logits = [torch.subtract(head(logits), 1 - self.action_mask[:, i], alpha = 1e5) 
                            for (i, head) in enumerate(self.actor_heads)]
            distributions = [Categorical(logits = logits) for logits in action_logits]

            actions = torch.stack([dist.sample().detach() for dist in distributions])
            actions_logprobs = torch.stack([dist.log_prob(action).detach() for (dist, action) in zip(distributions, actions)])
            state_values = self.critic(state).detach()

        return actions.T, actions_logprobs.T, state_values

    def evaluate(self, state : torch.Tensor, actions : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        distributions = [Categorical(logits = logits) for logits in action_logits] 

        actions_logprobs = torch.stack([dist.log_prob(action) for (dist, action) in zip(distributions, actions.T)]).T
        distribution_entropies = torch.stack([dist.entropy() for dist in distributions]).T
        state_values = self.critic(state)
        
        return actions_logprobs, state_values, distribution_entropies

    def get_actor_params(self):
        return list(self.actor_layers.parameters()) + list(self.actor_heads.parameters())