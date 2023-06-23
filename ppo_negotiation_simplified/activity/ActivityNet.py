import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from Actor import Actor
from Critic import Critic

from gym.spaces import MultiDiscrete

from typing import Tuple

class ActivityNet(nn.Module):
    def __init__(self, state_space: int, action_space : MultiDiscrete, params : dict = None):
        
        super().__init__()
        self.actor = Actor(state_space, action_space, params['actor'])
        self.critic = Critic(state_space, action_space, params['critic'])
        
    def act_deterministically(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action_logits = self.actor(env_state, action_mask)
        return torch.stack(list(map(torch.argmax, action_logits))).detach().cpu().numpy()
    
    def act_stochastically(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action_logits = self.actor(env_state, action_mask)
            distributions = [Categorical(logits = logits) for logits in action_logits]
        return torch.cat([d.sample() for d in distributions]).detach().cpu().numpy()

    def act(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():

            action_logits = self.actor(env_state, action_mask)
            state_values = self.critic(env_state, action_mask)

            distributions = [Categorical(logits = logits) for logits in action_logits]
            actions = torch.stack([dist.sample().detach() for dist in distributions])
            actions_logprobs = torch.stack([dist.log_prob(action).detach() for (dist, action) in zip(distributions, actions)])

        return actions.T, actions_logprobs.T, state_values

    def evaluate(self, 
                 env_state: torch.Tensor, 
                 action_mask: torch.Tensor, 
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        action_logits = self.actor(env_state, action_mask)
        state_values = self.critic(env_state, action_mask)

        distributions = [Categorical(logits = logits) for logits in action_logits]
        logprobs = torch.stack([dist.log_prob(action) for (dist, action) in zip(distributions, actions.T)]).T
        distribution_entropies = torch.stack([dist.entropy() for dist in distributions]).T
        
        return logprobs, state_values, distribution_entropies