import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from collections import deque
from tqdm import tqdm

################################## PPO Policy ##################################

_ACTION_MASK = 'action_mask'
_FEATURES = 'features'

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]


class ActorCritic(nn.Module):
    def __init__(self, state_space, action_space):
        super(ActorCritic, self).__init__()

        # actor
        self.actor_layers = nn.Sequential(
                        nn.Linear(state_space, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64)
                    )
        self.actor_heads = nn.ModuleList([
            nn.Linear(64, space.n)
            for space in action_space
        ])

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_space, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        concatenated_action_logits = torch.cat(action_logits)
        distributions = [Categorical(logits = logits) for logits in concatenated_action_logits]

        actions = torch.cat([dist.sample().detach().unsqueeze(0) for dist in distributions])
        actions_logprobs = torch.cat([dist.log_prob(action).detach().unsqueeze(0) for (dist, action) in zip(distributions, actions)])
        state_val = self.critic(state).detach()

        return actions, actions_logprobs, state_val

    def evaluate(self, state, actions):
        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        concatenated_action_logits = torch.cat(action_logits)
        distributions = [Categorical(logits = logits) for logits in concatenated_action_logits]

        actions_logprobs = torch.cat([dist.log_prob(action).unsqueeze(0) for (dist, action) in zip(distributions, actions)])
        distribution_entropies = torch.cat([dist.entropy().unsqueeze(0) for dist in distributions])
        state_values = self.critic(state)

        #return actions_logprobs, state_values, distribution_entropies
        return actions_logprobs, state_values, distribution_entropies.unsqueeze(0)

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor = .01, lr_critic = .01, gamma = .9, K_epochs = 20, eps_clip = .2):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim)
        actor_parameters = list(self.policy.actor_layers.parameters()) + list(self.policy.actor_heads.parameters())
        self.optimizer = torch.optim.Adam([
                        {'params': actor_parameters, 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def _get_flattened_obs(self, state) -> torch.Tensor:
        return torch.FloatTensor(state[_FEATURES].reshape(1, -1))

    def select_action(self, state):

        with torch.no_grad():
            state = self._get_flattened_obs(state)
            actions, actions_logprobs, state_val = self.policy_old.act(state)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(actions_logprobs)
        self.buffer.state_values.append(state_val)

        return np.array([action.item() for action in actions])

    def update(self):
        # Monte Carlo estimate of returns
        returns = deque()
        for reward in self.buffer.rewards[::-1]:
            discounted_return = returns[0] if len(returns) > 0 else 0
            returns.appendleft(self.gamma * discounted_return + reward)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()
        # calculate advantages
        advantages = returns.detach() - old_state_values.detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = zip(*[self.policy.evaluate(old_state.unsqueeze(0), old_action.unsqueeze(0)) 
                                                         for (old_state, old_action) in zip(old_states, old_actions)])
            logprobs, state_values, dist_entropy = torch.cat(logprobs), torch.cat(state_values), torch.cat(dist_entropy)
            #logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            #ratios = torch.exp(logprobs - old_logprobs.detach())
            ratios = torch.exp(logprobs - old_logprobs.detach()).T
            

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy.T#?
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
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
        
        
       


