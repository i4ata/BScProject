import torch
import torch.nn as nn
import numpy as np

from Interfaces import ActorCritic

import sys
sys.path.append("..")
from rice import Rice

from typing import Tuple, Dict, List

class NegotiationNet(ActorCritic):
    
    def __init__(self, env: Rice, n_features: int, params: dict = None):

        super(NegotiationNet, self).__init__()

        self.state_space = n_features + 2 * (env.num_agents - 1) * env.len_actions
        
        self.actor_layers = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.actor_heads = nn.ModuleList([
            nn.ModuleDict({
                'decision' : nn.Sequential(
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                ),
                'proposal' : nn.Sequential(
                    nn.Linear(64, env.len_actions),
                    nn.Sigmoid()
                ),
                'promise' : nn.Sequential(
                    nn.Linear(64, env.len_actions),
                    nn.Sigmoid()
                )
            })
            for agent in range(env.num_agents - 1)
        ])

        self.critic = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, env_state : torch.Tensor, **kwargs) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        with torch.no_grad():

            proposals_state = kwargs['proposals']
            promises_state = kwargs['promises']

            negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)

            decision_probs, proposal_probs, promise_probs = self._get_probs(negotiation_state)
            
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

        decision_probs, proposal_probs, promise_probs = self._get_probs(negotiation_state)

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
        return list(self.actor_layers.parameters()) + list(self.actor_heads.parameters())
            
    def _get_probs(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_layers(state)
        decision_probs, proposal_probs, promise_probs = zip(
            *[(head['decision'](logits), head['proposal'](logits), head['promise'](logits)) 
            for head in self.actor_heads]
        )

        decision_probs = torch.stack(decision_probs, dim = 1)
        proposal_probs = torch.stack(proposal_probs, dim = 1)
        promise_probs =  torch.stack(promise_probs, dim = 1)


        return decision_probs, proposal_probs, promise_probs