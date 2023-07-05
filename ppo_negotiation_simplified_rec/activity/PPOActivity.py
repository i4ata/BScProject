import torch
import torch.nn as nn
import numpy as np
from gym.spaces import MultiDiscrete

from activity.ActivityNet import ActivityNet

from typing import List, Optional, Tuple
import yaml

class RolloutBuffer:
    def __init__(self):

        self.actions: Optional[List[List[torch.Tensor]]] = None
        self.logprobs: Optional[List[List[torch.Tensor]]] = None
        self.action_masks: Optional[List[List[torch.Tensor]]] = None
        self.env_states: Optional[List[List[torch.Tensor]]] = None
        self.rewards: Optional[List[List[torch.Tensor]]] = None
        self.state_values: Optional[List[List[torch.Tensor]]] = None
        self.is_terminals: Optional[List[List[torch.Tensor]]] = None
        
        self.hidden_states_actor: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None
        self.hidden_states_critic: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None

    def clear(self):
        self.__init__()

class PPOActivity():
    
    def __init__(self, state_space: int, action_space: MultiDiscrete, device: str = 'cpu'):
        
        with open('activity/params.yml') as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
        self.device = device
        self.buffer = RolloutBuffer()
        self.policy = ActivityNet(state_space, action_space, self.params['policy']).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.params['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.params['lr_critic']}
                    ])

        self.policy_old = ActivityNet(state_space, action_space, self.params['policy']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> List[np.ndarray]:
            
        actions, logprobs, state_val = self.policy_old.act(env_state, action_mask)


        # If the buffer is empty, infer the batch size
        hs_actor = list(zip(*self.policy_old.actor.hidden_state))
        hs_critic = list(zip(*self.policy_old.critic.hidden_state))
        batch_size_iter = range(len(env_state))
        if self.buffer.env_states is None:
            self.buffer.env_states = [[env_state[i]] for i in batch_size_iter]
            self.buffer.action_masks = [[action_mask[i]] for i in batch_size_iter]
            self.buffer.actions = [[actions[i]] for i in batch_size_iter]
            self.buffer.logprobs = [[logprobs[i]] for i in batch_size_iter]
            self.buffer.state_values = [[state_val[i]] for i in batch_size_iter]
            self.buffer.rewards = [[] for i in batch_size_iter]
            self.buffer.is_terminals = [[] for i in batch_size_iter]

            self.buffer.hidden_states_actor = [[hs_actor[i]] for i in batch_size_iter]
            self.buffer.hidden_states_critic = [[hs_critic[i]] for i in batch_size_iter]
        else:
            for i in batch_size_iter:
                self.buffer.env_states[i].append(env_state[i])
                self.buffer.action_masks[i].append(action_mask[i])
                self.buffer.actions[i].append(actions[i])
                self.buffer.logprobs[i].append(logprobs[i])
                self.buffer.state_values[i].append(state_val[i])

                self.buffer.hidden_states_actor[i].append(hs_actor[i])
                self.buffer.hidden_states_critic[i].append(hs_critic[i])

        return [action.cpu().numpy() for action in actions]
    
    def update(self):
        
        # Monte Carlo estimate of returns
        returns = []
        discounted_return = 0
        # cat -> map -> stack gives the values in sequential and chronological order
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
        old_states = torch.cat(list(map(torch.stack, self.buffer.env_states))).detach()
        old_actions = torch.cat(list(map(torch.stack, self.buffer.actions))).detach()
        old_logprobs = torch.cat(list(map(torch.stack, self.buffer.logprobs))).detach()
        old_state_values = torch.cat(list(map(torch.stack, self.buffer.state_values))).detach()
        old_action_masks = torch.cat(list(map(torch.stack, self.buffer.action_masks))).detach()

        old_hidden_states_actor = self.to_tuple(self.buffer.hidden_states_actor)
        old_hidden_states_critic = self.to_tuple(self.buffer.hidden_states_critic)

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze().detach()).unsqueeze(-1)

        # Optimize policy for K epochs
        for _ in range(self.params['K_epochs']):
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                env_state=old_states, 
                action_mask=old_action_masks, 
                actions=old_actions,
                actor_hidden_state=old_hidden_states_actor,
                critic_hidden_state=old_hidden_states_critic
            )
            
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

    def to_tuple(self, xs: List[List[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = [item for sublist in xs for item in sublist]
        return torch.stack([hs[0] for hs in hidden_states]).detach(), \
               torch.stack([hs[1] for hs in hidden_states]).detach()