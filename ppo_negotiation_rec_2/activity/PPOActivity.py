import torch
import torch.nn as nn
import numpy as np
from gym.spaces import MultiDiscrete

from ActivityNet import ActivityNet

from typing import List

class RolloutBuffer:
    def __init__(self):

        self.actions = []
        self.logprobs = []

        self.action_masks = []

        self.env_states = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):

        del self.actions[:]
        del self.logprobs[:]
        del self.action_masks[:]
        del self.env_states[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPOActivity():
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, params: dict, device: str):
        
        self.params = params
        self.device = device
        self.buffer = RolloutBuffer()

        self.policy = ActivityNet(state_space, action_space, params['policy']).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': params['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': params['lr_critic']}
                    ])

        self.policy_old = ActivityNet(state_space, action_space, params['policy']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> List[np.ndarray]:
            
        actions, actions_logprobs, state_val = self.policy_old.act(env_state, action_mask)
        
        self.buffer.    env_states.     extend(env_state)
        self.buffer.    action_masks.   extend(action_mask)
        self.buffer.    actions.        extend(actions)
        self.buffer.    logprobs.       extend(actions_logprobs)
        self.buffer.    state_values.   extend(state_val)

        return [action.cpu().numpy() for action in actions]
    
    def update(self):
        
        # Monte Carlo estimate of returns
        returns = []
        discounted_return = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + self.params['gamma'] * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states          = torch.stack(self.buffer.env_states)        .detach()
        old_actions         = torch.stack(self.buffer.actions)           .detach()
        old_logprobs        = torch.stack(self.buffer.logprobs)          .detach()
        old_state_values    = torch.stack(self.buffer.state_values)      .detach()
        old_action_masks    = torch.stack(self.buffer.action_masks)      .detach()

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze().detach()).unsqueeze(-1)

        # Optimize policy for K epochs
        for _ in range(self.params['K_epochs']):
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_action_masks, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantages

            # final loss of clipped objective PPO
            loss = - torch.min(surr1, surr2) \
                   + self.params['mse_coef'] * self.MseLoss(state_values, returns) \
                   - self.params['entropy_coef'] * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))