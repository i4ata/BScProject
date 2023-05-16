from PPO_negotiation import PPO
from ActivityNet import ActivityNet
from NegotiationNet import NegotiationNet

import torch
import numpy as np

import sys
sys.path.append("..")
from rice import Rice

class Agent():
    def __init__(self, env : Rice, intial_state : dict, id : int, device : str = 'cpu'):
        self.nets = {
            'activityNet' : PPO(model = ActivityNet(env, len(intial_state['features'])), params = None, device = device),
            'negotiationNet' : PPO(model = NegotiationNet(env, len(intial_state['features'])), params = None, device = device)
        }

        self.device = device
        self.id = id

    # The functions below currently work for 1 state only (not with a batch of states)
    # Later remove the unsqueeze

    def act(self, state : dict):
        features = torch.FloatTensor(state['features'])
        action_mask = torch.FloatTensor(state['action_mask'].flatten())
        return self.nets['activityNet'].select_action(torch.cat((features, action_mask)).to(self.device).unsqueeze(0))
    
    def negotiate(self, state : dict):
        features = torch.FloatTensor(state['features'])
        promises = torch.FloatTensor(
            np.array(list(state['promises'].values())).flatten()
        )
        proposals = torch.FloatTensor(
            np.array(list(state['proposals'].values())).flatten()
        )
        return self.nets['negotiationNet'].select_action(torch.cat((features, promises, proposals)).to(self.device).unsqueeze(0))
    
    #

    def update(self):
        for net in self.nets:
            self.nets[net].update()