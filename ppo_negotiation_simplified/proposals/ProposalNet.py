import torch
import torch.nn as nn
import numpy as np

from proposals.Actor import Actor
from proposals.Critic import Critic

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

from typing import Tuple, Dict

class ProposalNet(nn.Module):
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, params: dict):

        super(ProposalNet, self).__init__()

        self.state_space = state_space
        self.action_space = action_space.nvec.sum()

        self.actor = Actor(self.state_space, self.action_space, params['actor'])
        self.critic = Critic(self.state_space, params['critic'])

    def act(self, env_state : torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        with torch.no_grad():

            proposal_probs, promise_probs = self.actor(env_state)
            state_value: torch.Tensor = self.critic(env_state)
            proposals = torch.bernoulli(proposal_probs)
            promises  = torch.bernoulli(promise_probs)

            log_probs_proposals = torch.log(proposals * proposal_probs + (1 - proposals) * (1 - proposal_probs))
            log_probs_promises = torch.log(promises * promise_probs + (1 - promises) * (1 - promise_probs))
            
            return_dict = {
                'proposals' : {
                    'proposals' : proposals,
                    'log_probs' : log_probs_proposals
                },
                'promises' : {
                    'promises' : promises,
                    'log_probs' : log_probs_promises
                }
            }

        return return_dict, state_value

    def evaluate(self, 
                 env_state: torch.Tensor, 
                 promises: torch.Tensor, 
                 proposals: torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        proposal_probs, promise_probs = self.actor(env_state)
        state_value = self.critic(env_state)

        proposal_log_probs = torch.log(proposals * proposal_probs + (1 - proposals) * (1 - proposal_probs))
        promise_log_probs = torch.log(promises * promise_probs + (1 - promises) * (1 - promise_probs))

        proposal_entropies = - proposal_probs * torch.log2(proposal_probs) - (1 - proposal_probs) * torch.log2(1 - proposal_probs)
        promise_entropies = - promise_probs * torch.log2(promise_probs) - (1 - promise_probs) * torch.log2(1 - promise_probs)

        return_dict = {
            'proposals' : {
                'log_probs' : proposal_log_probs,
                'entropies' : proposal_entropies
            },
            'promises' : {
                'log_probs' : promise_log_probs,
                'entropies' : promise_entropies
            }
        }

        return return_dict, state_value

    def eval_act(self, env_state: torch.Tensor, deterministic = False) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            proposal_probs, promise_probs = self.actor(env_state)
            
            if deterministic:
                proposals = torch.round(proposal_probs)
                promises = torch.round(promise_probs)
            else:
                proposals = torch.bernoulli(proposal_probs)
                promises = torch.bernoulli(promise_probs)
        return {'proposals': proposals.detach().cpu().numpy(), 'promises': promises.detach().cpu().numpy()}