import torch
import torch.nn as nn
from torch.distributions import Categorical

import copy

################################## PPO Policy ##################################

_FEATURES = 'features'
_ACTION_MASK = 'action_mask'

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        pass
        
    def forward(self):
        raise NotImplementedError
    
    def act_deterministically(self, state):
        pass
    
    def act_stochastically(self, state):
        pass

    def act(self, state, mask):
        pass

    def evaluate(self, state, actions):
        pass

    def get_actor_parameters(self):
        pass

class PPO:
    def __init__(self, model : ActorCritic, params : dict, device : str):

        self.params = params
        self.device = device

        self.buffer = RolloutBuffer()

        self.policy = model.to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.get_actor_parameters(), 'lr': .001},
                        {'params': self.policy.critic.parameters(), 'lr': .001}
                    ])

        self.policy_old = copy.deepcopy(self.policy)
        
        self.MseLoss = nn.MSELoss()

        self.loss_collection = []

    def select_action(self, states):
        """
        Select action in state `state` and fill in the rollout buffer with the relevant data
        """
            
        actions, actions_logprobs, state_val = self.policy_old.act(states)

        self.buffer.states.extend(states)
        self.buffer.actions.extend(actions)
        self.buffer.logprobs.extend(actions_logprobs)
        self.buffer.state_values.extend(state_val)

        return [action.cpu().numpy() for action in actions]
        #return actions.cpu().numpy()
    
    def update(self):
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
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        # calculate advantages
        advantages = (returns.detach() - old_state_values.detach()).unsqueeze(1)

        # Optimize policy for K epochs
        for _ in range(10):
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, .8, 1.2) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + .5 * self.MseLoss(state_values, returns) - .01 * dist_entropy
            
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
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))