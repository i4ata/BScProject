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
        
        self.activity_net = PPOActivity(state_space, action_space, device=device)
        self.proposal_net = PPOProposals(state_space, action_space, n_agents, device=device)
        self.decision_net = PPODecisions(state_space, action_space, n_agents, device=device)
        
        self.device = device
        self.id = id

    def make_proposals(self, states: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:

        negotiation_state = torch.FloatTensor(np.stack([state['features'] for state in states])).to(self.device)
        return self.proposal_net.select_action(negotiation_state)
    
    def make_decisions(self, states: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        
        negotiation_state = torch.FloatTensor(np.stack([
            np.concatenate([
                state['features'], 
                state['promises'].flatten(),
                state['proposals'].flatten(),
                state['own_promises'].flatten(),
                state['own_proposals'].flatten()
            ]) 
            for state in states
        ])).to(self.device)
        
        return self.decision_net.select_action(negotiation_state)

    def act(self, states : List[Dict[str, np.ndarray]]) -> List[np.ndarray]:

        features = torch.FloatTensor(np.stack([state['features'] for state in states])).to(self.device)
        action_mask = torch.FloatTensor(np.stack([state['action_mask'].flatten() for state in states])).to(self.device)
        
        return self.activity_net.select_action(features, action_mask)

    def update(self, nego_on = True) -> None:
        self.activity_net.update()
        if nego_on:
            self.proposal_net.update()
            self.decision_net.update()

    def eval_make_decisions(self, state: Dict[str, np.ndarray], deterministic = False) -> np.ndarray:

        negotiation_state = torch.FloatTensor(np.concatenate([
            state['features'], 
            state['promises'].flatten(),
            state['proposals'].flatten(),
            state['own_promises'].flatten(),
            state['own_proposals'].flatten()
        ])).unsqueeze(0).to(self.device)
        
        actions = self.decision_net.policy.eval_act(negotiation_state, deterministic)

        return actions

    def eval_make_proposals(self, state: Dict[str, np.ndarray], deterministic = False) -> Dict[str, np.ndarray]:

        negotiation_state = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        actions = self.proposal_net.policy.eval_act(negotiation_state, deterministic)
        
        return actions
    
    def eval_act(self, state: Dict[str, np.ndarray], deterministic = False) -> np.ndarray:

        features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(state['action_mask'].flatten()).unsqueeze(0).to(self.device)
        actions = self.activity_net.policy.eval_act(features, action_mask, deterministic)

        return actions