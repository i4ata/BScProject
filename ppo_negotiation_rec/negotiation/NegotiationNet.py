import torch
import torch.nn as nn
import numpy as np

from Interfaces import ActorCritic
from negotiation.Actor import Actor
from negotiation.Critic import Critic

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

from typing import Tuple, Dict, List

class NegotiationNet(ActorCritic):
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int, message_length: int, params: dict = None):

        super(NegotiationNet, self).__init__()

        self.state_space = state_space + 2 * (n_agents - 1) * action_space.nvec.sum() # + message_length * (n_agents - 1)
        self.action_space = action_space.nvec.sum()

        self.actor = Actor(self.state_space, self.action_space, n_agents, message_length)
        self.critic = Critic(self.state_space)


    def act(self, env_state : torch.Tensor, **kwargs) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        with torch.no_grad():

            proposals_state = kwargs['proposals']
            promises_state = kwargs['promises']
            # messages_state = kwargs['messages']

            #negotiation_state = torch.cat((env_state, proposals_state, promises_state, messages_state), dim = 1).to(env_state.device)
            negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)


            # decision_probs, proposal_probs, promise_probs, message_logits = self.actor(negotiation_state)
            
            decision_probs, proposal_probs, promise_probs = self.actor(negotiation_state)
            
            state_value = self.critic(negotiation_state)

            decisions = (torch.rand(decision_probs.shape).to(decision_probs.device) < decision_probs) * 1
            proposals = (torch.rand(proposal_probs.shape).to(proposal_probs.device) < proposal_probs) * 1
            promises  = (torch.rand(promise_probs .shape).to(promise_probs .device) < promise_probs ) * 1

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
                },
                # 'messages' : message_logits
            }

        return return_dict, state_value

    def evaluate(self, env_state: torch.Tensor, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        
        decisions = kwargs['actions']['decisions']
        proposals = kwargs['actions']['proposals']
        promises  = kwargs['actions']['promises' ]
        # messages_proposals = kwargs['actions']['messages']['proposals']
        # messages_decisions = kwargs['actions']['messages']['decisions']

        proposals_state = kwargs['states']['proposals']
        promises_state  = kwargs['states']['promises' ]

        proposals_state2 = kwargs['states']['proposals2']
        promises_state2 = kwargs['states']['promises2']
        
        # messages_state_decisions = kwargs['messages']['decisions']
        # messages_state_proposals = kwargs['messages']['proposals']

        # negotiation_state1 = torch.cat((env_state, proposals_state, promises_state, messages_state_decisions), dim = 1).to(env_state.device)
        # negotiation_state2 = torch.cat((env_state, proposals_state2, promises_state2, messages_state_proposals), dim = 1).to(env_state.device)

        negotiation_state1 = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)
        negotiation_state2 = torch.cat((env_state, proposals_state2, promises_state2), dim = 1).to(env_state.device)


        self.actor.hidden_state = kwargs['hidden_states']['proposals']['actor']
        self.critic.hidden_state = kwargs['hidden_states']['proposals']['critic']
        # _, proposal_probs, promise_probs, messages_proposals_logits = self.actor(negotiation_state2)
        _, proposal_probs, promise_probs = self.actor(negotiation_state2)
        state_value_proposal = self.critic(negotiation_state1)

        self.actor.hidden_state = kwargs['hidden_states']['decisions']['actor']
        self.critic.hidden_state = kwargs['hidden_states']['decisions']['critic']
        # decision_probs, _, _, messages_decisions_logits = self.actor(negotiation_state1)
        decision_probs, _, _ = self.actor(negotiation_state1)
        state_value_decision = self.critic(negotiation_state2)

        decision_log_probs = torch.log(torch.abs(decisions - decision_probs))
        proposal_log_probs = torch.log(torch.abs(proposals - proposal_probs))
        promise_log_probs  = torch.log(torch.abs(promises  - promise_probs))

        decision_entropies = - decision_probs * torch.log2(decision_probs) - (1 - decision_probs) * torch.log2(1 - decision_probs)
        proposal_entropies = - proposal_probs * torch.log2(proposal_probs) - (1 - proposal_probs) * torch.log2(1 - proposal_probs)
        promise_entropies  = - promise_probs  * torch.log2(promise_probs ) - (1 - promise_probs ) * torch.log2(1 - promise_probs)

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
            },
            'state_values' : {
                'decision' : state_value_decision,
                'proposal' : state_value_proposal
            }
        }

        return return_dict
    
    def act_deterministically(self, env_state: torch.Tensor, **kwargs) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            
            proposals_state = kwargs['proposals']
            promises_state = kwargs['promises']

            negotiation_state = torch.cat((env_state, proposals_state, promises_state), dim = 1).to(env_state.device)

            decision_probs, proposal_probs, promise_probs = self.actor(negotiation_state)
            
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

            decision_probs, proposal_probs, promise_probs = self.actor(negotiation_state)
            
            decisions = ((torch.rand(decision_probs.shape) < decision_probs.cpu()) * 1).detach().numpy()
            proposals = ((torch.rand(proposal_probs.shape) < proposal_probs.cpu()) * 1).detach().numpy()
            promises  = ((torch.rand(promise_probs.shape ) < promise_probs .cpu()) * 1).detach().numpy()

            return_dict = {
                'decisions': decisions,
                'proposals': proposals,
                'promise': promises
            }
        return return_dict

    def get_actor_parameters(self) -> List[nn.parameter.Parameter]:
        return self.actor.parameters()