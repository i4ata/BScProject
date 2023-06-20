from negotiation.decisions.PPODecisions import PPODecisions
from negotiation.decisions.DecisionNet import DecisionNet

from negotiation.proposals.PPOProposals import PPOProposals
from negotiation.proposals.ProposalNet import ProposalNet

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
                model = ActivityNet(state_space, action_space), 
                params = None, device = device
            ),
            'proposalNet' : PPOProposals(
                model = ProposalNet(state_space, action_space, n_agents), 
                params = None, device = device
            ),
            'decisionNet' : PPODecisions(
                model = DecisionNet(state_space, action_space, n_agents),
                params = None, device = device
            )
        }

        self.device = device
        self.id = id

    def make_proposals(self, states: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:

        # Create a 2D tensor of the features only [batch_size, features]
        # Pass it to the proposal network and receive the promises and proposals
        features = torch.FloatTensor(np.stack([
            np.append(state['features'], state['negotiation_status'])
            for state in states
        ])).to(self.device)
        
        return self.nets['proposalNet'].select_action(features)
    
    def make_decisions(self, states: List[Dict[str, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        
        features = torch.FloatTensor(np.stack([
            np.append(state['features'], state['negotiation_status']) 
            for state in states
        ])).to(self.device)
        promises = torch.FloatTensor(np.stack([
            state['promises'].flatten()
            for state in states
        ])).to(self.device)
        proposals = torch.FloatTensor(np.stack([
            state['proposals'].flatten()
            for state in states
        ])).to(self.device)

        return self.nets['decisionNet'].select_action(
            features,
            proposals = proposals,
            promises = promises
        )

    def act(self, states : List[Dict[str, np.ndarray]]) -> List[np.ndarray]:

        features = torch.FloatTensor(np.stack([state['features'] for state in states])).to(self.device)
        
        action_mask = torch.FloatTensor(np.stack([state['action_mask'].flatten() for state in states])).to(self.device)
        
        return self.nets['activityNet'].select_action(
            env_state = features,
            action_mask = action_mask,
        )

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
        
        for net in ('proposalNet', 'decisionNet'):

            self.nets[net].policy_old.actor.hidden_state = None
            self.nets[net].policy_old.critic.hidden_state = None

            self.nets[net].policy.actor.hidden_state = None
            self.nets[net].policy.critic.hidden_state = None