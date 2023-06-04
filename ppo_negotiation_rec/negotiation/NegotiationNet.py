import torch
import torch.nn as nn
import numpy as np

from Interfaces import ActorCritic
from negotiation.Actor import Actor
from negotiation.Critic import Critic

import sys
sys.path.append("..")
from rice import Rice

from typing import Tuple, Dict, List

class NegotiationNet(ActorCritic):
    
    def __init__(self, env: Rice, n_features: int, params: dict = None):

        super(NegotiationNet, self).__init__()

        self.state_space = n_features + 2 * (env.num_agents - 1) * env.len_actions

        self.actor = Actor(state_space=self.state_space, action_space=env.len_actions, num_agents=env.num_agents)
        self.critic = Critic(state_space=self.state_space)


    def act(self, env_state : torch.Tensor, **kwargs) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        with torch.no_grad():

            proposals_state = kwargs['proposals']
            promises_state = kwargs['promises']

            negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)

            decision_probs, proposal_probs, promise_probs = self.actor(negotiation_state)
            
            state_value = self.critic(negotiation_state)

            decisions = (torch.rand(decision_probs.shape).to(decision_probs.device) < decision_probs) * 1
            proposals = (torch.rand(proposal_probs.shape).to(proposal_probs.device) < proposal_probs) * 1
            promises  = (torch.rand(promise_probs.shape).to(promise_probs.device) < promise_probs) * 1

            log_probs_decisions = torch.log(torch.abs(decisions - decision_probs))
            log_probs_proposals = torch.log(torch.abs(proposals - proposal_probs))
            log_probs_promise   = torch.log(torch.abs(promises  - promise_probs))
            
            return_dict = {
                'decisions' : {
                    'decisions' : decisions,
                    'log_probs' : log_probs_decisions
                },
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
        
        decisions = kwargs['actions']['decisions']
        proposals = kwargs['actions']['proposals']
        promises = kwargs['actions']['promises']

        proposals_state = kwargs['states']['proposals']
        promises_state = kwargs['states']['promises']

        negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)

        decision_probs, proposal_probs, promise_probs = self.actor(negotiation_state)

        state_value = self.critic(negotiation_state)

        decision_log_probs = torch.log(torch.abs(decisions - decision_probs))
        proposal_log_probs = torch.log(torch.abs(proposals - proposal_probs))
        promise_log_probs = torch.log(torch.abs(promises - promise_probs))

        decision_entropies = - decision_probs * torch.log2(decision_probs) - (1 - decision_probs) * torch.log2(1 - decision_probs)
        proposal_entropies = - proposal_probs * torch.log2(proposal_probs) - (1 - proposal_probs) * torch.log2(1 - proposal_probs)
        promise_entropies = - promise_probs * torch.log2(promise_probs) - (1 - promise_probs) * torch.log2(1 - promise_probs)

        return_dict = {
            'decisions' : {
                'log_probs' : decision_log_probs,
                'entropies' : decision_entropies,
            },
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
    
    def act_deterministically(self, env_state: torch.Tensor, **kwargs) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            
            proposals_state = kwargs['proposals']
            promises_state = kwargs['promises']

            negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)

            decision_probs, proposal_probs, promise_probs = self._get_probs(negotiation_state)
            
            decisions = ((decision_probs > .5) * 1) .detach().cpu().numpy()
            proposals = ((proposal_probs > .5) * 1) .detach().cpu().numpy()
            promises  = ((promise_probs  > .5) * 1) .detach().cpu().numpy()

            return_dict = {
                'decisions': decisions,
                'proposals': proposals,
                'promise': promises
            }
        return return_dict
    
    def act_stochastically(self, env_state: torch.Tensor, **kwargs) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            
            proposals_state = kwargs['proposals']
            promises_state = kwargs['promises']

            negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)

            decision_probs, proposal_probs, promise_probs = self._get_probs(negotiation_state)
            
            decisions = (
                (torch.rand(decision_probs.shape).to(decision_probs.device) < decision_probs) * 1
            ).detach().cpu().numpy()
            proposals = (
                (torch.rand(proposal_probs.shape).to(proposal_probs.device) < proposal_probs) * 1
            ).detach().cpu().numpy()
            promises  = (
                (torch.rand(promise_probs.shape).to(promise_probs.device) < promise_probs) * 1
            ).detach().cpu().numpy()

            return_dict = {
                'decisions': decisions,
                'proposals': proposals,
                'promise': promises
            }
        return return_dict

    def get_actor_parameters(self) -> List[nn.parameter.Parameter]:
        return self.actor.parameters()
    
    def reset(self):
        self.actor.hidden_state = None
        self.critic.hidden_state = None