from decisions.PPODecisions import PPODecisions
from proposals.PPOProposals import PPOProposals
from activity.PPOActivity import PPOActivity

import torch
import numpy as np

from typing import Dict, List

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

class Agent():
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int, id : int, device : str = 'cpu'):
        self.nets: dict = {
            'activityNet' : PPOActivity(state_space, action_space, device = device),
            'proposalNet' : PPOProposals(state_space, action_space, n_agents, device = device),
            'decisionNet' : PPODecisions(state_space, action_space, n_agents, device = device)
        }

        self.device = device
        self.id = id

    def make_proposals(self, states: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:

        negotiation_state = torch.FloatTensor(np.stack([state['features'] for state in states])).to(self.device)
        return self.nets['proposalNet'].select_action(negotiation_state)
    
    def make_decisions(self, states: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        
        negotiation_state = torch.FloatTensor(np.stack([
            np.concatenate([
                state['features'], 
                state['promises'].flatten(),
                state['proposals'].flatten()
            ]) 
            for state in states
        ])).to(self.device)
        
        return self.nets['decisionNet'].select_action(negotiation_state)

    def act(self, states : List[Dict[str, np.ndarray]]) -> List[np.ndarray]:

        features = torch.FloatTensor(np.stack([state['features'] for state in states])).to(self.device)
        action_mask = torch.FloatTensor(np.stack([state['action_mask'].flatten() for state in states])).to(self.device)
        
        return self.nets['activityNet'].select_action(features, action_mask)

    def update(self, nego_on = True) -> None:
        if nego_on:
            for net in self.nets:
                self.nets[net].update()
        else:
            self.nets['activityNet'].update()

    def eval_make_decisions(self, state: Dict[str, np.ndarray], deterministic = False) -> np.ndarray:

        negotiation_state = torch.FloatTensor(np.concatenate([
            state['features'], 
            state['promises'].flatten(),
            state['proposals'].flatten()])
        ).unsqueeze(0).to(self.device)

        actions = self.nets['decisionNet'].policy.act_stochastically(negotiation_state)

        return actions

    def eval_make_proposals(self, state: Dict[str, np.ndarray], deterministic = False) -> Dict[str, np.ndarray]:

        negotiation_state = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)

        actions = self.nets['proposalNet'].policy.act_stochastically(negotiation_state)
        
        return actions
    
    def eval_act(self, state: Dict[str, np.ndarray], deterministic = False) -> np.ndarray:
        features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(state['action_mask'].flatten()).unsqueeze(0).to(self.device)

        actions = self.nets['activityNet'].policy.act_stochastically(
            env_state = features,
            action_mask = action_mask
        )

        return actions