import torch
import torch.nn as nn
import numpy as np

from Interfaces import ActorCritic
from negotiation.decisions.Actor import Actor
from negotiation.decisions.Critic import Critic

import sys
sys.path.append("..")
from gym.spaces import MultiDiscrete

from typing import Tuple, Dict, List

class DecisionNet(ActorCritic):
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int):

        super(DecisionNet, self).__init__()

        features = state_space
        proposals = (n_agents - 1) * action_space.nvec.sum()
        promises = (n_agents - 1) * action_space.nvec.sum()
        negotiation_status = n_agents - 1
        own_mask = action_space.nvec.sum()

        self.state_space = sum((features, proposals, promises, negotiation_status, own_mask))
        self.action_space = 1

        self.actor = Actor(self.state_space, n_agents)
        self.critic = Critic(self.state_space)

    def act(self, env_state : torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():

            # Create the state by concatenating the envirnoment state and the current decisions and proposals
            negotiation_state = torch.cat(
                (env_state, kwargs['proposals'], kwargs['promises']), dim = 1
            ).to(env_state.device)
            
            # Forward pass the state to the actor and critic
            decision_probs: torch.Tensor = self.actor(negotiation_state)
            state_value: torch.Tensor = self.critic(negotiation_state)

            # Sample from the decision probabilities' distributions
            decisions = (torch.rand(decision_probs.shape).to(decision_probs.device) < decision_probs) * 1
            
            # Get the log probabilities of the samples
            log_probs = torch.log(torch.abs(decisions - decision_probs))

        return decisions, log_probs, state_value

    def evaluate(self, env_state: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Create the negotiation state by concatenating the environment state with the current proposals and promises
        negotiation_state = torch.cat(
            (env_state, kwargs['state_proposals'], kwargs['state_promises']), dim = 1
        ).to(env_state.device)

        # Forward pass the state to the actor and critic
        decision_probs = self.actor(negotiation_state, hidden_state = kwargs['hidden_states_actor'])
        state_value = self.critic(negotiation_state, hidden_state = kwargs['hidden_states_critic'])

        # Get log probs
        log_probs = torch.log(torch.abs(kwargs['decisions'] - decision_probs))

        # Compute distribution entropies
        entropies = - decision_probs * torch.log2(decision_probs) - (1 - decision_probs) * torch.log2(1 - decision_probs)

        return log_probs, entropies, state_value
    
    def act_deterministically(self, env_state: torch.Tensor, **kwargs) -> np.ndarray:
        with torch.no_grad():

            # Create state
            negotiation_state = torch.cat(
                (env_state, kwargs['proposals'], kwargs['promises']), dim = 1
            ).to(env_state.device)

            # Get probabilities by passing state to actor
            decision_probs: torch.Tensor = self.actor(negotiation_state)
            
            # Round probabilities to get decisiosn
            decisions = ((decision_probs > .5) * 1).cpu().numpy()
        
        return decisions
    
    def act_stochastically(self, env_state: torch.Tensor, **kwargs) -> np.ndarray:
        with torch.no_grad():
            
            # Create state
            negotiation_state = torch.cat(
                (env_state, kwargs['proposals'], kwargs['promises']), dim = 1
            ).to(env_state.device)

            # Get probabilities by passing state to actor
            decision_probs: torch.Tensor = self.actor(negotiation_state)
            
            # Sample from the probability
            decisions = ((torch.rand_like(decision_probs) < decision_probs.cpu()) * 1).numpy()
            
        return decisions

    def get_actor_parameters(self) -> List[nn.parameter.Parameter]:
        return self.actor.parameters()