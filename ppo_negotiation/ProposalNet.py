import torch
import torch.nn as nn
import numpy as np

from PPO_negotiation import ActorCritic

import sys
sys.path.append("..")
from rice import Rice

from typing import Tuple

class ProposalNet(ActorCritic):
    def __init__(self, env : Rice, n_features: int, params : dict = None):

        super(ProposalNet, self).__init__()

        self.state_space = n_features
        self.action_space = env.num_agents * env.len_actions

        self.actor = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space),
            nn.Sigmoid()
        )

        self.critic = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        raise NotImplementedError
    
    def act(self, state : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            probs = self.actor(state)
            state_value = self.critic(state)
            actions = (torch.rand(probs.shape).to(probs.device) < probs) * 1
            log_probs = torch.log(torch.abs(actions - probs))

        return actions, log_probs, state_value

    def evaluate(self, state : torch.Tensor, actions : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.actor(state)
        state_value = self.critic(state)
        log_probs = torch.log(torch.abs(actions - probs))
        
        entropies = - probs * torch.log2(probs) - (1 - probs) * torch.log2(1 - probs)
        
        return log_probs, state_value, entropies
    
    def act_deterministically(self, state : torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = self.actor(state)
        return ((probs > .5) * 1).detach().cpu().numpy()
    
    def act_stochastically(self, state : torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = self.actor(state)
        return ((torch.rand(probs.shape).to(probs.device) < probs) * 1).detach().cpu().numpy()
        
    def get_actor_parameters(self):
        return self.actor.parameters()