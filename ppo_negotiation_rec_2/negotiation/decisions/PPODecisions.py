import torch
from Interfaces import PPO, ActorCritic

from typing import List, Tuple
import numpy as np

################################## PPO Policy ##################################

class PPODecisions(PPO):
    def __init__(self, model: ActorCritic, params: dict, device: str):
        super().__init__(model, params, device)
    

    def select_action(self, env_state, **kwargs) -> List[np.ndarray]:

        # Pass the state to the policy and get the decisions, the logprobs, and the state value
        (decisions, log_probs, state_val) : Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = \
              self.policy_old.act(env_state, kwargs)

        # Save in memory
        self.buffer.    env_states.             extend(env_state)
        self.buffer.    proposals_states.       extend(kwargs['proposals'])
        self.buffer.    promises_states.        extend(kwargs['promises'])
        self.buffer.    decisions.              extend(decisions)
        self.buffer.    decisions_logprobs.     extend(log_probs)
        self.buffer.    state_values.           extend(state_val)
        self.buffer.    hidden_states_actor.    extend(zip(*self.policy_old.actor.hidden_state))
        self.buffer.    hidden_states_critic.   extend(zip(*self.policy_old.critic.hidden_state))
        
        # return decisions as numpy arrays
        return [decision.detach().cpu().numpy() for decision in decisions]
    
    def update(self) -> None:
        """
        Update the policy with PPO
        """

        # Monte Carlo estimate of returns
        returns = []
        discounted_return = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + .9 * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states                = torch.stack(self.buffer.env_states)           .detach()
        old_decisions             = torch.stack(self.buffer.decisions)            .detach()
        old_decisions_logprobs    = torch.stack(self.buffer.decisions_logprobs)   .detach()
        old_proposals_states      = torch.stack(self.buffer.proposals_states)     .detach()
        old_promises_states       = torch.stack(self.buffer.promises_states)      .detach()
        old_state_values          = torch.stack(self.buffer.state_values)         .detach()

        old_hidden_states_actor   = self.to_tuple(self.buffer.hidden_states_actor)
        old_hidden_states_critic  = self.to_tuple(self.buffer.hidden_states_critic)

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze()).unsqueeze(-1).unsqueeze(-1)

        # Optimize policy for K epochs
        for epoch in range(10):
            # Evaluating old actions and values
            (log_probs, entropies, state_values): Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.policy.evaluate(
                old_states,
                decisions = old_decisions,
                state_proposals = old_proposals_states,
                state_promises = old_promises_states,
                hidden_states_actor = old_hidden_states_actor,
                hidden_states_critic = old_hidden_states_critic
            )
            
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_decisions = torch.exp(log_probs - old_decisions_logprobs.detach())
            
            # Finding Surrogate Loss
            surr1_decisions = ratios_decisions * advantages
            surr2_decisions = torch.clamp(ratios_decisions, .8, 1.2) * advantages

            # final loss of clipped objective PPO
            loss: torch.Tensor = - torch.min(surr1_decisions, surr2_decisions) \
                                 + .5 * self.MseLoss(state_values, returns) \
                                 - .01 * entropies
            
            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            self.loss_collection.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    # Transform a list of 1d hidden states to 1 2d hidden state
    def to_tuple(self, xs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.stack([x[0] for x in xs]).detach(), torch.stack([x[1] for x in xs]).detach())
