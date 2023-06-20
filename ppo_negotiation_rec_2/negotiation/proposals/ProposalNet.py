import torch
import torch.nn as nn
import numpy as np

from Interfaces import ActorCritic
from negotiation.proposals.Actor import Actor
from negotiation.proposals.Critic import Critic

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

from typing import Tuple, Dict, List

class ProposalNet(ActorCritic):
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int):

        super(ProposalNet, self).__init__()

        features = state_space
        negotiation_status = n_agents - 1
        own_mask = action_space.nvec.sum()

        self.state_space = sum((features, negotiation_status, own_mask))
        self.action_space = action_space.nvec.sum()

        self.actor = Actor(self.state_space, self.action_space, n_agents)
        self.critic = Critic(self.state_space)


    def act(self, env_state : torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        with torch.no_grad():

            proposal_probs, promise_probs = self.actor(env_state)
            state_value: torch.Tensor = self.critic(env_state)

            proposals = (torch.rand(proposal_probs.shape).to(proposal_probs.device) < proposal_probs) * 1
            promises  = (torch.rand(promise_probs .shape).to(promise_probs .device) < promise_probs ) * 1

            log_probs_proposals = torch.log(torch.abs(proposals - proposal_probs))
            log_probs_promise   = torch.log(torch.abs(promises  - promise_probs))
            
            return_dict = {
                'proposals' : {
                    'proposals' : proposals,
                    'log_probs' : log_probs_proposals
                },
                'promises' : {
                    'promises' : promises,
                    'log_probs' : log_probs_promise
                }
            }

        return return_dict, state_value

    def evaluate(self, env_state: torch.Tensor, **kwargs) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        proposal_probs, promise_probs = self.actor(env_state, kwargs['hidden_states_actor'])
        state_value = self.critic(env_state, kwargs['hidden_states_critic'])

        proposal_log_probs = torch.log(torch.abs(kwargs['proposals'] - proposal_probs))
        promise_log_probs  = torch.log(torch.abs(kwargs['promises']  - promise_probs))

        proposal_entropies = - proposal_probs * torch.log2(proposal_probs) - (1 - proposal_probs) * torch.log2(1 - proposal_probs)
        promise_entropies  = - promise_probs  * torch.log2(promise_probs ) - (1 - promise_probs ) * torch.log2(1 - promise_probs)

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
    
    def act_deterministically(self, env_state: torch.Tensor) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            
            proposal_probs, promise_probs = self.actor(env_state)
            
            proposals = ((proposal_probs > .5) * 1) .detach().cpu().numpy()
            promises  = ((promise_probs  > .5) * 1) .detach().cpu().numpy()

        return {'proposals' : proposals, 'promises' : promises}
    
    def act_stochastically(self, env_state: torch.Tensor) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            
            proposal_probs, promise_probs = self.actor(env_state)

            proposals = ((torch.rand(proposal_probs.shape) < proposal_probs.cpu()) * 1).detach().numpy()
            promises  = ((torch.rand(promise_probs.shape ) < promise_probs .cpu()) * 1).detach().numpy()

        return {'proposals' : proposals, 'promises' : promises}

    def get_actor_parameters(self) -> List[nn.parameter.Parameter]:
        return self.actor.parameters()