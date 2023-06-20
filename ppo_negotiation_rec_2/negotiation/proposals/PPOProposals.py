import torch
from Interfaces import PPO, ActorCritic

from typing import Dict, List, Tuple
import numpy as np

################################## PPO Policy ##################################

class PPOProposals(PPO):
    def __init__(self, model: ActorCritic, params: dict, device: str):
        super().__init__(model, params, device)
    

    def select_action(self, env_state) -> Dict[str, List[np.ndarray]]:
        """
        Select action in state `state` and fill in the rollout buffer with the relevant data
        """

        actions, state_val = self.policy_old.act(env_state)

        self.buffer.    env_states.             extend(env_state)
        self.buffer.    proposals.              extend(actions['proposals']['proposals'])
        self.buffer.    proposals_logprobs.     extend(actions['proposals']['log_probs'])
        self.buffer.    promises.               extend(actions['promises']['promises'])
        self.buffer.    promises_logprobs.      extend(actions['promises']['log_probs'])
        self.buffer.    state_values.           extend(state_val)
        self.buffer.    hidden_states_actor.    extend(zip(*self.policy_old.actor.hidden_state))
        self.buffer.    hidden_states_critic.   extend(zip(*self.policy_old.critic.hidden_state))
            

        return {
            'proposals' : [proposal.detach().cpu().numpy() for proposal in actions['proposals']['proposals']],
            'promises' : [promise.detach().cpu().numpy() for promise in actions['promises']['promises']]
        }
    
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
        old_states                  = torch.stack(self.buffer.env_states)           .detach()
        
        old_proposals               = torch.stack(self.buffer.proposals)            .detach()
        old_proposals_logprobs      = torch.stack(self.buffer.proposals_logprobs)   .detach()
        old_promises                = torch.stack(self.buffer.promises)             .detach()
        old_promises_logprobs       = torch.stack(self.buffer.promises_logprobs)    .detach()
        
        old_state_values            = torch.stack(self.buffer.state_values)         .detach()

        old_hidden_states_actor     = self.to_tuple(self.buffer.hidden_states_actor)
        old_hidden_states_critic    = self.to_tuple(self.buffer.hidden_states_critic)

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze()).unsqueeze(-1).unsqueeze(-1)
                    

        # Optimize policy for K epochs
        for _ in range(10):
            # Evaluating old actions and values
            actions, state_val  = self.policy.evaluate(
                old_states, 
                proposals = old_proposals,
                promises = old_promises,
                hidden_states_actor = old_hidden_states_actor,
                hidden_states_critic = old_hidden_states_critic
            )
            
            state_values = torch.squeeze(state_val)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_proposals = torch.exp(actions['proposals']['log_probs'] - old_proposals_logprobs.detach())
            ratios_promises  = torch.exp(actions['promises' ]['log_probs'] - old_promises_logprobs .detach())
            
            # Finding Surrogate Loss
            surr1_proposals = ratios_proposals * advantages
            surr2_proposals = torch.clamp(ratios_proposals, .8, 1.2) * advantages

            surr1_promises  = ratios_promises  * advantages
            surr2_promises  = torch.clamp(ratios_promises,  .8, 1.2) * advantages

            # final loss of clipped objective PPO
            loss = - torch.min(surr1_proposals, surr2_proposals) \
                   - torch.min(surr1_promises,  surr2_promises) \
                   + .5 * self.MseLoss(state_values, returns) \
                   - .01 * actions['promises']['entropies'] \
                   - .01 * actions['proposals']['entropies']
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
