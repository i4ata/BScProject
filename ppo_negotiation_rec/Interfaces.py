import torch
import torch.nn as nn
import copy

from typing import List
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.decisions = []
        self.decisions_logprobs = []

        self.proposals = []
        self.proposals_logprobs = []

        self.promises = []
        self.promises_logprobs = []

        self.actions = []
        self.logprobs = []

        self.action_masks = []

        self.env_states = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

        self.proposals_states = []
        self.promises_states = []

        self.hidden_states_proposals_actor = []
        self.hidden_states_proposals_critic = []

        self.hidden_states_decisions_actor = []
        self.hidden_states_decisions_critic = []

        self.rewards_proposals = []
        self.rewards_decisions = []

        self.state_values_decisions = []
        self.state_values_proposals = []
    
    def clear(self):
        del self.decisions[:]
        del self.decisions_logprobs[:]
        del self.proposals[:]
        del self.proposals_logprobs[:]
        del self.promises[:]
        del self.promises_logprobs[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.action_masks[:]
        del self.env_states[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

        del self.proposals_states[:]
        del self.promises_states[:]
        del self.hidden_states_proposals_actor[:]
        del self.hidden_states_proposals_critic[:]
        del self.hidden_states_decisions_actor[:]
        del self.hidden_states_decisions_critic[:]
        del self.rewards_proposals[:]
        del self.rewards_decisions[:]

        del self.state_values_decisions[:]
        del self.state_values_proposals[:]

        

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        pass
        
    def forward(self):
        raise NotImplementedError
    
    def act_deterministically(self, env_state, **kwargs):
        pass
    
    def act_stochastically(self, env_state, **kwargs):
        pass

    def act(self, env_state, **kwargs):
        pass

    def evaluate(self, env_state, **kwargs):
        pass

    def get_actor_parameters(self):
        pass

class PPO:
    def __init__(self, model : ActorCritic, params : dict, device : str):

        self.params = params
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = model.to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.get_actor_parameters(), 'lr': .001},
                        {'params': self.policy.critic.parameters(), 'lr': .001}
                    ])

        self.policy_old = copy.deepcopy(self.policy)
        
        self.MseLoss = nn.MSELoss()

        self.loss_collection = []

    def select_action(self, env_state, **kwargs) -> List[np.ndarray]:
        pass
    
    def update(self):
        pass
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))