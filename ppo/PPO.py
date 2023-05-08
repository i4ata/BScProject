import torch
import torch.nn as nn
from torch.distributions import Categorical


################################## PPO Policy ##################################

_FEATURES = 'features'

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
    def __init__(self, state_space, action_space, params):
        super(ActorCritic, self).__init__()

        # actor
        self.actor_layers = [nn.Linear(state_space, params['hidden_size_actor'])]
        for actor_layer in range(params['n_layers_actor']):
            self.actor_layers.extend([
                nn.Linear(params['hidden_size_actor'], params['hidden_size_actor']),
                nn.Tanh()
            ])
        self.actor_layers = nn.Sequential(*self.actor_layers)
        self.actor_heads = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], space.n)
            for space in action_space
        ])

        # critic
        self.critic = [nn.Linear(state_space, params['hidden_size_critic'])]
        for critic_layer in range(params['n_layers_critic']):
            self.critic.extend([
                nn.Linear(params['hidden_size_critic'], params['hidden_size_critic']),
                nn.Tanh()
            ])
        self.critic.append(nn.Linear(params['hidden_size_critic'], 1))
        self.critic = nn.Sequential(*self.critic)
        
    def forward(self):
        raise NotImplementedError
    
    def act_deterministically(self, state):
        """
        Take the best action in the `state`. `state` needs to be a single state instead of a batch of states.
        This function is used to evaluate the model by making the policy deterministic.
        """
        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        return torch.stack(list(map(torch.argmax, action_logits))).detach().cpu().numpy()
    
    def act_stochastically(self, state):
        """
        Take an action in the `state`. `state` needs to be a single state instead of a batch of states.
        This function is used to evaluate the model.
        """
        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        distributions = [Categorical(logits = logits) for logits in action_logits]
        return torch.cat([d.sample() for d in distributions]).detach().cpu().numpy()

    def act(self, state):
        """
        Take actions in `state`.
        Returns the actions (1 per environment action space), the logprobs (1 per action), and the state value (1)
        """
        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        distributions = [Categorical(logits = logits) for logits in action_logits]

        actions = torch.stack([dist.sample().detach() for dist in distributions])
        actions_logprobs = torch.stack([dist.log_prob(action).detach() for (dist, action) in zip(distributions, actions)])
        state_values = self.critic(state).detach()

        return actions.T, actions_logprobs.T, state_values

    def evaluate(self, state, actions):
        """
        Evaluate the action `actions` in state `state`.
        Returns the logprobs (1 per the action taken in each environment action space), the state value (1) and the entropies of the distributions (1 per action space)
        """
        logits = self.actor_layers(state)
        action_logits = [head(logits) for head in self.actor_heads]
        distributions = [Categorical(logits = logits) for logits in action_logits] 

        actions_logprobs = torch.stack([dist.log_prob(action) for (dist, action) in zip(distributions, actions.T)]).T
        distribution_entropies = torch.stack([dist.entropy() for dist in distributions]).T
        state_values = self.critic(state)
        
        return actions_logprobs, state_values, distribution_entropies

class PPO:
    def __init__(self, state_dim, action_dim, params, device):

        self.params = params
        self.device = device
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, params).to(device)
        actor_parameters = list(self.policy.actor_layers.parameters()) + list(self.policy.actor_heads.parameters())
        self.optimizer = torch.optim.Adam([
                        {'params': actor_parameters, 'lr': params['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': params['lr_critic']}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, params).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.loss_collection = []

    def select_action(self, states):
        """
        Select action in state `state` and fill in the rollout buffer with the relevant data
        """
        if not isinstance(states, list):
            states = [states]

        with torch.no_grad():
            states = torch.stack(list(map(lambda state: torch.FloatTensor(state[_FEATURES]), states))).to(self.device)
            actions, actions_logprobs, state_val = self.policy_old.act(states)

        self.buffer.states.extend(states)
        self.buffer.actions.extend(actions)
        self.buffer.logprobs.extend(actions_logprobs)
        self.buffer.state_values.extend(state_val)

        return [action.cpu().numpy() for action in actions]
    
    def select_action_deterministically(self, state):
        """
        Act deterministically in state `state`
        """
        with torch.no_grad():
            state = torch.FloatTensor(state[_FEATURES].reshape(1, -1)).to(self.device)
            actions = self.policy.act_deterministically(state)
        return actions

    def select_action_stochastically(self, state):
        """
        Act stochastically in state `state`
        """
        with torch.no_grad():
            state = torch.FloatTensor(state[_FEATURES].reshape(1, -1)).to(self.device)
            actions = self.policy.act_stochastically(state)
        return actions
    
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
            discounted_return = reward + (self.params['gamma'] * discounted_return)
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()
        print(old_states.shape)
        # calculate advantages
        advantages = (returns.detach() - old_state_values.detach()).unsqueeze(1)

        # Optimize policy for K epochs
        for _ in range(self.params['K_epochs']):
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.params['eps_clip'], 1 + self.params['eps_clip']) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + self.params['critic_loss_parameter'] * self.MseLoss(state_values, returns) - self.params['entropy_parameter'] * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            #loss = (loss ** 2).mean()
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
        
        
       


