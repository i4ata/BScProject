import torch
from Interfaces import PPO, ActorCritic

from typing import Dict, List, Tuple
import numpy as np

################################## PPO Policy ##################################

class PPONegotiation(PPO):
    def __init__(self, model: ActorCritic, params: dict, device: str):
        super().__init__(model, params, device)
    

    def select_action(self, states, save = True) -> Dict[str, List[np.ndarray]]:
        """
        Select action in state `state` and fill in the rollout buffer with the relevant data
        """

        actions, state_val = self.policy_old.act(states)

        return_dict = {
            'decisions' : [decision.cpu().numpy() for decision in actions['decisions']['decisions']],
            'proposals' : [proposal.cpu().numpy() for proposal in actions['proposals']['proposals']],
            'promise' : [promise.cpu().numpy() for promise in actions['promises']['promises']]
        }

        if not save:
            return return_dict

        self.buffer.states.extend(states)
        self.buffer.state_values.extend(state_val)

        self.buffer.decisions.extend(actions['decisions']['decisions'])
        self.buffer.decisions_logprobs.extend(actions['decisions']['log_probs'])

        self.buffer.proposals.extend(actions['proposals']['proposals'])
        self.buffer.proposals_logprobs.extend(actions['proposals']['log_probs'])

        self.buffer.promises.extend(actions['promises']['promises'])
        self.buffer.promises_logprobs.extend(actions['promises']['log_probs'])

        return return_dict
    
    def update(self) -> None:
        """
        Update the policy with PPO
        """
        # Monte Carlo estimate of returns
        returns = []
        discounted_return = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + .9 * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()
        old_decisions = torch.squeeze(torch.stack(self.buffer.decisions, dim = 0)).detach()
        old_decisions_logprobs = torch.squeeze(torch.stack(self.buffer.decisions_logprobs, dim = 0)).detach()
        old_proposals = torch.squeeze(torch.stack(self.buffer.proposals, dim = 0)).detach()
        old_proposals_logprobs = torch.squeeze(torch.stack(self.buffer.proposals_logprobs, dim = 0)).detach()
        old_promises = torch.squeeze(torch.stack(self.buffer.promises, dim = 0)).detach()
        old_promises_logprobs = torch.squeeze(torch.stack(self.buffer.promises_logprobs, dim = 0)).detach()
        

        # calculate advantages
        # it's necessary to make the advantages 3-dimensional since the decisions and proposals are also 3 dimensional:
        # [batch_size x num_agents x decision / proposal]
        # Since there is one decision per agent (0 or 1), that 3rd dimension is gone from the squeezing so I unsqueeze manually later
        advantages = (returns.detach() - old_state_values.detach()).unsqueeze(1).unsqueeze(1)

        # Optimize policy for K epochs
        for _ in range(10):
            
            # Evaluating old actions and values
            actions, state_values = self.policy.evaluate(
                old_states, 
                old_decisions.unsqueeze(-1), 
                old_proposals,
                old_promises) # look at the unsqueeze
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_decisions = torch.exp(actions['decisions']['log_probs'] - old_decisions_logprobs.unsqueeze(-1).detach()) # look at the unsqueeze
            ratios_proposals = torch.exp(actions['proposals']['log_probs'] - old_proposals_logprobs.detach())
            ratios_promises = torch.exp(actions['promises']['log_probs'] - old_promises_logprobs.detach())
            
            
            # Finding Surrogate Loss
            surr1_decisions = ratios_decisions * advantages
            surr2_decisions = torch.clamp(ratios_decisions, .8, 1.2) * advantages

            surr1_proposals = ratios_proposals * advantages
            surr2_proposals = torch.clamp(ratios_proposals, .8, 1.2) * advantages

            surr1_promises = ratios_promises * advantages
            surr2_promises = torch.clamp(ratios_promises, .8, 1.2) * advantages


            # final loss of clipped objective PPO
            loss = -torch.min(surr1_proposals, surr2_proposals) - \
                    torch.min(surr1_decisions, surr2_decisions) - \
                    torch.min(surr1_promises, surr2_promises) + \
                    .5 * self.MseLoss(state_values, returns) - \
                    .01 * actions['decisions']['entropies'] - \
                    .01 * actions['proposals']['entropies'] - \
                    .01 * actions['promises']['entropies']
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