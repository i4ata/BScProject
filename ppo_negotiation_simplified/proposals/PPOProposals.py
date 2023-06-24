import torch
import torch.nn as nn
from gym.spaces import MultiDiscrete
import numpy as np
import yaml

from proposals.ProposalNet import ProposalNet

from typing import Dict, List, Optional

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):

        self.proposals: Optional[List[List[torch.Tensor]]] = None
        self.proposals_logprobs: Optional[List[List[torch.Tensor]]] = None
        self.promises: Optional[List[List[torch.Tensor]]] = None
        self.promises_logprobs: Optional[List[List[torch.Tensor]]] = None
        self.env_states: Optional[List[List[torch.Tensor]]] = None
        self.rewards: Optional[List[List[torch.Tensor]]] = None
        self.state_values: Optional[List[List[torch.Tensor]]] = None
        self.is_terminals: Optional[List[List[torch.Tensor]]] = None

    def clear(self):
        self.__init__()


class PPOProposals():
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int, device: str):
        
        with open('proposals/params.yml') as f:
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

        batch_size_iter = range(len(env_state))
        if self.buffer.env_states is None:
            self.buffer.env_states = [[env_state[i]] for i in batch_size_iter]

            self.buffer.proposals = [[actions['proposals']['proposals'][i]] for i in batch_size_iter]
            self.buffer.proposals_logprobs = [[actions['proposals']['log_probs'][i]] for i in batch_size_iter]
            
            self.buffer.promises = [[actions['promises']['promises'][i]] for i in batch_size_iter]
            self.buffer.promises_logprobs = [[actions['promises']['log_probs'][i]] for i in batch_size_iter]

            self.buffer.state_values = [[state_val[i]] for i in batch_size_iter]
            self.buffer.rewards = [[] for i in batch_size_iter]
            self.buffer.is_terminals = [[] for i in batch_size_iter]
        else:
            for i in batch_size_iter:
                self.buffer.env_states[i].append(env_state[i])

                self.buffer.proposals[i].append(actions['proposals']['proposals'][i])
                self.buffer.proposals_logprobs[i].append(actions['proposals']['log_probs'][i])

                self.buffer.promises[i].append(actions['promises']['promises'][i])
                self.buffer.promises_logprobs[i].append(actions['promises']['log_probs'][i])

                self.buffer.state_values[i].append(state_val[i])

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
        zipped = zip(
            reversed(np.concatenate(list(map(np.stack, self.buffer.rewards)))), 
            reversed(np.concatenate(list(map(np.stack, self.buffer.is_terminals))))
        )
        for reward, is_terminal in zipped:
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + self.params['gamma'] * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states                = torch.cat(list(map(torch.stack, self.buffer.env_states))).detach()
        old_proposals             = torch.cat(list(map(torch.stack, self.buffer.proposals))).detach()
        old_proposals_logprobs    = torch.cat(list(map(torch.stack, self.buffer.proposals_logprobs))).detach()
        old_promises              = torch.cat(list(map(torch.stack, self.buffer.promises))).detach()
        old_promises_logprobs     = torch.cat(list(map(torch.stack, self.buffer.promises_logprobs))).detach()
        old_state_values          = torch.cat(list(map(torch.stack, self.buffer.state_values))).detach()

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