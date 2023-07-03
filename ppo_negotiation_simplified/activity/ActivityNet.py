import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from gym.spaces import MultiDiscrete

from typing import Tuple, List

class Actor(nn.Module):

    def __init__(self, state_space: int, action_space: MultiDiscrete, params: dict = None):
        super().__init__()

        self.input_layer = nn.Linear(state_space, params['hidden_size_actor'])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], params['hidden_size_actor']) 
            for i in range(params['n_hidden_layers_actor'])])
        self.activation = nn.Tanh()
        self.flatten_mask = nn.Flatten()

        self.heads = nn.ModuleList([
            nn.Linear(params['hidden_size_actor'], space.n)
            for space in action_space
        ])

    def forward(self, env_state: torch.Tensor, action_mask: torch.Tensor) -> List[torch.Tensor]:

        x = self.activation(self.input_layer(torch.cat((env_state, self.flatten_mask(action_mask)), dim = 1)))
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        action_logits = [torch.subtract(head(x), 1 - action_mask[:, i], alpha = 1e5)
                         for (i, head) in enumerate(self.heads)]
        return action_logits

class Critic(nn.Module):

    def __init__(self, state_space: int, params):
        super().__init__()

        self.input_layer = nn.Linear(state_space, params['hidden_size_critic'])
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(params['hidden_size_critic'], params['hidden_size_critic']) 
            for i in range(params['n_hidden_layers_critic'])])
        self.output_layer = nn.Linear(params['hidden_size_critic'], 1)
        self.activation = nn.Tanh()
        self.flatten_mask = nn.Flatten()

    def forward(self, env_state: torch.Tensor, action_mask = torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(torch.cat((env_state, self.flatten_mask(action_mask)), dim = 1)))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

class ActivityNet(nn.Module):

    def __init__(self, state_space: int, action_space : MultiDiscrete, params : dict = None):
        
        super().__init__()
        self.actor = Actor(state_space, action_space, params['actor'])
        self.critic = Critic(state_space, params['critic'])
        
    def eval_act(self, env_state: torch.Tensor, action_mask: torch.Tensor, deterministic = False) -> np.ndarray:
        with torch.no_grad():
            action_logits = self.actor(env_state, action_mask)
            if deterministic:
                actions = torch.stack(list(map(torch.argmax, action_logits)))
            else:
                distributions = [Categorical(logits=logits) for logits in action_logits]
                actions = torch.cat([d.sample() for d in distributions])

        return actions.detach().cpu().numpy()
    
    def act(self, 
            env_state: torch.Tensor, 
            action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():

            action_logits = self.actor(env_state, action_mask)
            state_values = self.critic(env_state, action_mask)
            distributions = [Categorical(logits = logits) for logits in action_logits]
            actions = torch.stack([dist.sample().detach() for dist in distributions])
            actions_logprobs = torch.stack([dist.log_prob(action).detach() for dist, action in zip(distributions, actions)])

        return actions.T, actions_logprobs.T, state_values

    def evaluate(self, 
                 env_state: torch.Tensor, 
                 action_mask: torch.Tensor, 
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        action_logits = self.actor(env_state, action_mask)
        state_values = self.critic(env_state, action_mask)

        distributions = [Categorical(logits = logits) for logits in action_logits]
        logprobs = torch.stack([dist.log_prob(action) for (dist, action) in zip(distributions, actions.T)]).T
        distribution_entropies = torch.stack([dist.entropy() for dist in distributions]).T
        
        return logprobs, state_values, distribution_entropies