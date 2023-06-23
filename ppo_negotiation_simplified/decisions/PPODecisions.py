import torch
import torch.nn as nn
import numpy as np
from gym.spaces import MultiDiscrete
import yaml

from decisions.DecisionNet import DecisionNet

from typing import List, Tuple

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):

        self.actions = []
        self.logprobs = []
        self.env_states = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):

        del self.actions[:]
        del self.logprobs[:]
        del self.env_states[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PPODecisions():
    def __init__(self, state_space: int, action_space: MultiDiscrete, n_agents: int, device: str):
        
        with open('ppo_negotiation_simplified/decisions/params.yml') as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
        self.device = device
        self.buffer = RolloutBuffer()

        self.policy = DecisionNet(state_space, action_space, n_agents, self.params['policy']).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.params['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.params['lr_critic']}
                    ])

        self.policy_old = DecisionNet(state_space, action_space, n_agents, self.params['policy']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, env_state) -> List[np.ndarray]:

        decisions, log_probs, state_val = self.policy_old.act(env_state)

        self.buffer.env_states.extend(env_state)
        self.buffer.state_values.extend(state_val)
        self.buffer.actions.extend(decisions)
        self.buffer.logprobs.extend(log_probs)

        return [decision.detach().cpu().numpy() for decision in decisions]
    
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

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states = torch.stack(self.buffer.env_states).detach()
        old_decisions = torch.stack(self.buffer.actions).detach()
        old_decisions_logprobs = torch.stack(self.buffer.logprobs).detach()
        old_state_values = torch.stack(self.buffer.state_values).detach()

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze()).unsqueeze(-1).unsqueeze(-1)

        # Optimize policy for K epochs
        for epoch in range(self.params['K_epochs']):
            # Evaluating old actions and values
            log_probs, entropies, state_values = self.policy.evaluate(old_states, old_decisions)
            
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_decisions = torch.exp(log_probs - old_decisions_logprobs.detach())
            
            # Finding Surrogate Loss
            surr1_decisions = ratios_decisions * advantages
            surr2_decisions = torch.clamp(ratios_decisions, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantages

            # final loss of clipped objective PPO
            loss: torch.Tensor = - torch.min(surr1_decisions, surr2_decisions) \
                                 + self.params['mse_coef'] * self.MseLoss(state_values, returns) \
                                 - self.params['entropy_coef'] * entropies
            
            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()