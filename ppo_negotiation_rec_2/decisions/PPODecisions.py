import torch
import torch.nn as nn
import yaml
from typing import List, Tuple, Optional
import numpy as np
from decisions.DecisionNet import DecisionNet
from gym.spaces import MultiDiscrete

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):
        
        self.actions: Optional[List[List[torch.Tensor]]] = None
        self.logprobs: Optional[List[List[torch.Tensor]]] = None
        self.env_states: Optional[List[List[torch.Tensor]]] = None
        self.rewards: Optional[List[List[torch.Tensor]]] = None
        self.state_values: Optional[List[List[torch.Tensor]]] = None
        self.is_terminals: Optional[List[List[torch.Tensor]]] = None

        self.hidden_states_actor: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None
        self.hidden_states_critic: Optional[List[List[Tuple[torch.Tensor, torch.Tensor]]]] = None

    def clear(self):
        self.__init__()

class PPODecisions:
    def __init__(self, state_space: int, action_space: MultiDiscrete, device: str):
        
        with open('decisions/params.yml') as f:
            self.params = yaml.load(f, Loader=yaml.FullLoader)
        self.device = device
        self.buffer = RolloutBuffer()

        self.policy = DecisionNet(state_space, action_space, self.params['policy']).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.params['lr_actor']},
                        {'params': self.policy.critic.parameters(), 'lr': self.params['lr_critic']}
                    ])

        self.policy_old = DecisionNet(state_space, action_space, self.params['policy']).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, env_state: torch.Tensor) -> List[np.ndarray]:

        decisions, logprobs, state_val = self.policy_old.act(env_state)
        
        hs_actor = list(zip(*self.policy_old.actor.hidden_state))
        hs_critic = list(zip(*self.policy_old.critic.hidden_state))
        batch_size_iter = range(len(env_state))

        if self.buffer.env_states is None:
            self.buffer.env_states = [[env_state[i]] for i in batch_size_iter]
            self.buffer.actions = [[decisions[i]] for i in batch_size_iter]
            self.buffer.logprobs = [[logprobs[i]] for i in batch_size_iter]
            self.buffer.state_values = [[state_val[i]] for i in batch_size_iter]
            self.buffer.rewards = [[] for i in batch_size_iter]
            self.buffer.is_terminals = [[] for i in batch_size_iter]

            self.buffer.hidden_states_actor = [[hs_actor[i]] for i in batch_size_iter]
            self.buffer.hidden_states_critic = [[hs_critic[i]] for i in batch_size_iter]
        else:
            for i in batch_size_iter:
                self.buffer.env_states[i].append(env_state[i])
                self.buffer.actions[i].append(decisions[i])
                self.buffer.logprobs[i].append(logprobs[i])
                self.buffer.state_values[i].append(state_val[i])

                self.buffer.hidden_states_actor[i].append(hs_actor[i])
                self.buffer.hidden_states_critic[i].append(hs_critic[i])

        return [decision.detach().cpu().numpy() for decision in decisions]
    
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
            discounted_return = reward + .9 * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states = torch.cat(list(map(torch.stack, self.buffer.env_states))).detach()
        old_decisions = torch.cat(list(map(torch.stack, self.buffer.actions))).detach()
        old_decisions_logprobs = torch.cat(list(map(torch.stack, self.buffer.logprobs))).detach()
        old_state_values = torch.cat(list(map(torch.stack, self.buffer.state_values))).detach()

        old_hidden_states_actor   = self.to_tuple(self.buffer.hidden_states_actor)
        old_hidden_states_critic  = self.to_tuple(self.buffer.hidden_states_critic)
        
        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze()).unsqueeze(-1)

        # Optimize policy for K epochs
        for epoch in range(10):
            # Evaluating old actions and values
            log_probs, entropies, state_values = self.policy.evaluate(
                env_state=old_states,
                decisions=old_decisions,
                hidden_states_actor=old_hidden_states_actor,
                hidden_states_critic=old_hidden_states_critic
            )
            
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_decisions = torch.exp(log_probs - old_decisions_logprobs.detach())
            
            # Finding Surrogate Loss
            surr1_decisions = ratios_decisions * advantages
            surr2_decisions = torch.clamp(ratios_decisions, .8, 1.2) * advantages

            # final loss of clipped objective PPO
            loss: torch.Tensor = - torch.min(surr1_decisions, surr2_decisions) \
                                 + .5 * self.MseLoss(state_values, returns) \
                                 - .01 * entropies
            
            # take gradient step
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def to_tuple(self, xs: List[List[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = [item for sublist in xs for item in sublist]
        return torch.stack([hs[0] for hs in hidden_states]).detach(), \
               torch.stack([hs[1] for hs in hidden_states]).detach()
