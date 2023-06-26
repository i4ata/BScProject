import torch
import torch.nn as nn
import numpy as np

from decisions.Actor import Actor
from decisions.Critic import Critic

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

from typing import Tuple, Dict, List

class DecisionNet(nn.Module):
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, params: dict):

        super(DecisionNet, self).__init__()

        self.state_space = 4 * action_space.nvec.sum() + state_space
        self.action_space = 1

        self.actor = Actor(self.state_space, params['actor'])
        self.critic = Critic(self.state_space, params['critic'])

    def act(self, env_state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():

            decision_probs: torch.Tensor = self.actor(env_state)
            state_value: torch.Tensor = self.critic(env_state)
            decisions = torch.bernoulli(decision_probs)            
            log_probs = torch.log(decision_probs * decisions + (1 - decision_probs) * (1 - decisions))

        return decisions, log_probs, state_value

    def evaluate(self, env_state: torch.Tensor, decisions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        decision_probs = self.actor(env_state)
        state_value = self.critic(env_state)
        log_probs = torch.log(decision_probs * decisions + (1 - decision_probs) * (1 - decisions))
        entropies = - decision_probs * torch.log2(decision_probs) - (1 - decision_probs) * torch.log2(1 - decision_probs)
        return log_probs, entropies, state_value
    
    def eval_act(self, env_state: torch.Tensor, deterministic = False) -> np.ndarray:

        with torch.no_grad():
            decision_probs: torch.Tensor = self.actor(env_state)
            if deterministic:
                decisions = torch.round(decision_probs)
            else:
                decisions = torch.bernoulli(decision_probs)
            return decisions.cpu().numpy()