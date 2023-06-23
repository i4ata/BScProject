import torch
import torch.nn as nn
from gym.spaces import MultiDiscrete
import numpy as np
import yaml

from proposals.ProposalNet import ProposalNet

from typing import Dict, List

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):

        self.proposals = []
        self.proposals_logprobs = []
        self.promises = []
        self.promises_logprobs = []
        self.env_states = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):

        del self.proposals[:]
        del self.proposals_logprobs[:]
        del self.promises[:]
        del self.promises_logprobs[:]
        del self.env_states[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPOProposals():
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int, device: str):
        
        with open('ppo_negotiation_simplified/proposals/params.yml') as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
        self.device = device
        self.buffer = RolloutBuffer()

        self.policy = ProposalNet(state_space, action_space, n_agents, self.params['policy']).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.params['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.params['lr_critic']}
                    ])

        self.policy_old = ProposalNet(state_space, action_space, n_agents, self.params['policy']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

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
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + self.params['gamma'] * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states                  = torch.stack(self.buffer.env_states)           .detach()
        old_proposals               = torch.stack(self.buffer.proposals)            .detach()
        old_proposals_logprobs      = torch.stack(self.buffer.proposals_logprobs)   .detach()
        old_promises                = torch.stack(self.buffer.promises)             .detach()
        old_promises_logprobs       = torch.stack(self.buffer.promises_logprobs)    .detach()
        old_state_values            = torch.stack(self.buffer.state_values)         .detach()

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze()).unsqueeze(-1).unsqueeze(-1)

        # Optimize policy for K epochs
        for _ in range(self.params['K_epochs']):
            # Evaluating old actions and values
            actions, state_val  = self.policy.evaluate(old_states, old_promises, old_proposals)
            
            state_values = torch.squeeze(state_val)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_proposals = torch.exp(actions['proposals']['log_probs'] - old_proposals_logprobs.detach())
            ratios_promises  = torch.exp(actions['promises' ]['log_probs'] - old_promises_logprobs .detach())
            
            # Finding Surrogate Loss
            surr1_proposals = ratios_proposals * advantages
            surr2_proposals = torch.clamp(ratios_proposals, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantages

            surr1_promises  = ratios_promises  * advantages
            surr2_promises  = torch.clamp(ratios_promises, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantages

            # final loss of clipped objective PPO
            loss = - torch.min(surr1_proposals, surr2_proposals) \
                   - torch.min(surr1_promises,  surr2_promises) \
                   + self.params['mse_coef'] * self.MseLoss(state_values, returns) \
                   - self.params['entropy_coef'] * actions['promises']['entropies'] \
                   - self.params['entropy_coef'] * actions['proposals']['entropies']
            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()