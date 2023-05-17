from PPOActivity import PPOActivity
from PPONegotiation import PPONegotiation
from ActivityNet import ActivityNet
from NegotiationNet import NegotiationNet

import torch
import numpy as np

from typing import Dict, List

import sys
sys.path.append("..")
from rice import Rice

class Agent():
    def __init__(self, env : Rice, intial_state : dict, id : int, device : str = 'cpu'):
        self.nets = {
            'activityNet' : PPOActivity(model = ActivityNet(env, len(intial_state['features'])), params = None, device = device),
            'negotiationNet' : PPONegotiation(model = NegotiationNet(env, len(intial_state['features'])), params = None, device = device)
        }

        self.device = device
        self.id = id

    def act(self, states : List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        features = torch.FloatTensor(np.array([state['features'] for state in states]))
        action_mask = torch.FloatTensor(np.array([state['action_mask'].flatten() for state in states]))
        
        return self.nets['activityNet'].select_action(torch.cat((features, action_mask), dim = 1).to(self.device))
    
    def negotiate(self, states : List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        features = torch.FloatTensor(np.array([state['features'] for state in states]))
        
        promises = torch.FloatTensor(np.array([
            np.array(list(state['promises'].values())).flatten()
            for state in states
        ]))
        
        proposals = torch.FloatTensor(np.array([
            np.array(list(state['proposals'].values())).flatten()
            for state in states
        ]))
        
        return self.nets['negotiationNet'].select_action(torch.cat((features, promises, proposals), dim = 1).to(self.device))
    
    #

    def update(self) -> None:
        for net in self.nets:
            self.nets[net].update()