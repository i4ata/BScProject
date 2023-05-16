import torch
import torch.nn as nn
import numpy as np

from PPO_negotiation import ActorCritic

import sys
sys.path.append("..")
from rice import Rice

from typing import Tuple, Dict

class NegotiationNet(ActorCritic):
    
    def __init__(self, env: Rice, n_features: int, params: dict = None):

        super(NegotiationNet, self).__init__()

        self.state_space = n_features + (2 * env.num_agents - 1) * env.len_actions
        
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
                )
            })
            for agent in range(env.num_agents)
        ])

        self.critic = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, 
            state : torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        with torch.no_grad():
            logits = self.actor_layers(state)
            decision_probs, proposal_probs = zip(
                *[(head['decision'](logits), head['proposal'](logits)) 
                  for head in self.actor_heads]
            )

            state_value = self.critic(state)

            decision_probs, proposal_probs = torch.stack(decision_probs, dim = 1), torch.stack(proposal_probs, dim = 1)

            decisions = (torch.rand(decision_probs.shape).to(decision_probs.device) < decision_probs) * 1
            proposals = (torch.rand(proposal_probs.shape).to(proposal_probs.device) < proposal_probs) * 1

            log_probs_decisions = torch.log(torch.abs(decisions - decision_probs))
            log_probs_proposals = torch.log(torch.abs(proposals - proposal_probs))
            
            return_dict = {
                'decisions' : {
                    'decisions' : decisions,
                    'log_probs' : log_probs_decisions
                },
                'proposals' : {
                    'proposals' : proposals,
                    'log_probs' : log_probs_proposals
                }
            }

        return return_dict, state_value

    def evaluate(self, 
                 state: torch.Tensor, 
                 decisions: torch.Tensor, 
                 proposals: torch.Tensor) -> Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]:
        
        logits = self.actor_layers(state)
        decision_probs, proposal_probs = zip(
            *[(head['decision'](logits), head['proposal'](logits)) 
              for head in self.actor_heads]
        )
        decision_probs, proposal_probs = torch.stack(decision_probs, dim = 1), torch.stack(proposal_probs, dim = 1)
        
        state_value = self.critic(state)
        
        decision_log_probs = torch.log(torch.abs(decisions - decision_probs))
        proposal_log_probs = torch.log(torch.abs(proposals - proposal_probs))
        
        decision_entropies = - decision_probs * torch.log2(decision_probs) - (1 - decision_probs) * torch.log2(1 - decision_probs)
        proposal_entropies = - proposal_probs * torch.log2(proposal_probs) - (1 - proposal_probs) * torch.log2(1 - proposal_probs)

        return_dict = {
            'decisions' : {
                'log_probs' : decision_log_probs,
                'entropies' : decision_entropies,
            },
            'proposals' : {
                'log_probs' : proposal_log_probs,
                'entropies' : proposal_entropies
            }
        }

        return return_dict, state_value
    
    def act_deterministically(self, state: torch.Tensor) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            logits = self.actor_layers(state)
            decision_probs, proposal_probs = zip(
                *[(head['decision'](logits), head['proposal'](logits)) 
                  for head in self.actor_heads]
            )
            decision_probs, proposal_probs = torch.stack(decision_probs, dim = 1), torch.stack(proposal_probs, dim = 1)

            decisions = ((decision_probs > .5) * 1).detach().cpu().numpy()
            proposals = ((proposal_probs > .5) * 1).detach().cpu().numpy()
            
            return_dict = {
                'decisions': decisions,
                'proposals': proposals
            }
        return return_dict
    
    def act_stochastically(self, state: torch.Tensor) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            logits = self.actor_layers(state)
            decision_probs, proposal_probs = zip(
                *[(head['decision'](logits), head['proposal'](logits)) 
                  for head in self.actor_heads]
            )
            decision_probs, proposal_probs = torch.stack(decision_probs, dim = 1), torch.stack(proposal_probs, dim = 1)

            decisions = ((torch.rand(decision_probs.shape).to(decision_probs.device) < decision_probs) * 1).detach().cpu().numpy()
            proposals = ((torch.rand(proposal_probs.shape).to(proposal_probs.device) < proposal_probs) * 1).detach().cpu().numpy()

            return_dict = {
                'decisions': decisions,
                'proposals': proposals
            }
        return return_dict

    def get_actor_parameters(self):
        return list(self.actor_layers.parameters()) + list(self.actor_heads.parameters())
            