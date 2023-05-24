from PPOActivity import PPOActivity
from PPONegotiation import PPONegotiation
from ActivityNet import ActivityNet
from NegotiationNet import NegotiationNet
from Interfaces import PPO

import torch
import numpy as np

from typing import Dict, List

import sys
sys.path.append("..")
from rice import Rice

class Agent():
    def __init__(self, env : Rice, intial_state : dict, id : int, device : str = 'cpu'):
        self.nets: Dict[str, PPO] = {
            'activityNet' : PPOActivity(model = ActivityNet(env, len(intial_state['features'])), params = None, device = device),
            'negotiationNet' : PPONegotiation(model = NegotiationNet(env, len(intial_state['features'])), params = None, device = device)
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
    
    def negotiate(self, states: List[Dict[str, np.ndarray]], save_map: Dict[str, bool]) -> List[np.ndarray]:

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

        actions = self.nets['negotiationNet'].select_action(env_state = features, 
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