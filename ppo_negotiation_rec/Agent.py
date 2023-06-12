from negotiation.PPONegotiation import PPONegotiation
from negotiation.NegotiationNet import NegotiationNet

from activity.PPOActivity import PPOActivity
from activity.ActivityNet import ActivityNet

from Interfaces import PPO

import torch
import numpy as np

from typing import Dict, List

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

class Agent():
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int, id : int, device : str = 'cpu'):
        self.nets: Dict[str, PPO] = {
            'activityNet' : PPOActivity(
                model = ActivityNet(state_space=state_space, action_space=action_space), 
                params = None, device = device
            ),
            'negotiationNet' : PPONegotiation(
                model = NegotiationNet(state_space=state_space, action_space=action_space, n_agents=n_agents), 
                params = None, device = device
            )
        }

        self.device = device
        self.id = id

    def act(self, states : List[Dict[str, np.ndarray]], with_mask = True) -> List[np.ndarray]:

        features = torch.FloatTensor(np.array([state['features'] for state in states])).to(self.device)
        
        action_mask = torch.FloatTensor(np.array([state['action_mask'].flatten() for state in states])).to(self.device)
        
        if not with_mask:
            action_mask = torch.ones_like(action_mask)

        return self.nets['activityNet'].select_action(
            env_state = features,
            action_mask = action_mask,
            save = with_mask
        )
    
    def negotiate(self, 
                  states: List[Dict[str, np.ndarray]], 
                  save_map: Dict[str, bool]) -> List[np.ndarray]:

        features = torch.FloatTensor(np.array([state['features'] for state in states])).to(self.device)
        
        promises = torch.FloatTensor(np.array([
            np.array(list(state['promises'].values())).flatten()
            for state in states
        ])).to(self.device)
        
        proposals = torch.FloatTensor(np.array([
            np.array(list(state['proposals'].values())).flatten()
            for state in states
        ])).to(self.device)
        
        # tensor_state = torch.cat((features, promises, proposals), dim = 1).to(self.device)

        actions = self.nets['negotiationNet'].select_action(
            env_state = features, 
            promises = promises, 
            proposals = proposals,
            save_state = save_map['save_state'],
            save_decisions = save_map['save_decisions'],
            save_proposals_promises = save_map['save_proposals_promises']
        )

        return actions

    def update(self) -> None:
        for net in self.nets:
            self.nets[net].update()

    def eval_negotiate(self, state: Dict[str, np.ndarray], deterministic = True) -> Dict[str, np.ndarray]:
        features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        promises = torch.FloatTensor(np.array(list(state['promises'].values())).flatten()).unsqueeze(0).to(self.device)
        proposals = torch.FloatTensor(np.array(list(state['proposals'].values())).flatten()).unsqueeze(0).to(self.device)

        actions = self.nets['negotiationNet'].policy.act_deterministically(
            env_state = features,
            promises = promises,
            proposals = proposals
        ) if deterministic else self.nets['negotiationNet'].policy.act_stochastically(
            env_state = features,
            promises = promises,
            proposals = proposals
        )

        return actions
    
    def eval_act(self, state: Dict[str, np.ndarray], deterministic = True) -> np.ndarray:
        features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(state['action_mask'].flatten()).unsqueeze(0).to(self.device)

        actions = self.nets['activityNet'].policy.act_deterministically(
            env_state = features,
            action_mask = action_mask
        ) if deterministic else self.nets['activityNet'].policy.act_stochastically(
            env_state = features,
            action_mask = action_mask
        )

        return actions

    def reset_negotiation_hs(self):
        
        self.nets['negotiationNet'].policy_old.actor.hidden_state = None
        self.nets['negotiationNet'].policy_old.critic.hidden_state = None

        self.nets['negotiationNet'].policy.actor.hidden_state = None
        self.nets['negotiationNet'].policy.critic.hidden_state = None