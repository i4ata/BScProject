import torch
from Interfaces import ActorCritic, PPO

from typing import Tuple

class PPOActivity(PPO):
    
    def __init__(self, model: ActorCritic, params: dict, device: str):
        super().__init__(model, params, device)


    def select_action(self, env_state: torch.Tensor, **kwargs):
        """
        Select action in state `state` and fill in the rollout buffer with the relevant data
        """
            
        (actions, actions_logprobs, state_val): Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = \
            self.policy_old.act(env_state, **kwargs)
        
        self.buffer.    env_states.     extend(env_state)
        self.buffer.    action_masks.   extend(kwargs['action_mask'])
        self.buffer.    actions.        extend(actions)
        self.buffer.    logprobs.       extend(actions_logprobs)
        self.buffer.    state_values.   extend(state_val)

        return [action.cpu().numpy() for action in actions]
    
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
        old_states          = torch.stack(self.buffer.env_states)        .detach()
        old_actions         = torch.stack(self.buffer.actions)           .detach()
        old_logprobs        = torch.stack(self.buffer.logprobs)          .detach()
        old_state_values    = torch.stack(self.buffer.state_values)      .detach()
        old_action_masks    = torch.stack(self.buffer.action_masks)      .detach()

        # calculate advantages
        advantages = (returns.detach() - old_state_values.squeeze().detach()).unsqueeze(-1)

        # Optimize policy for K epochs
        for _ in range(10):
            
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(env_state=old_states, 
                                                                        actions=old_actions, 
                                                                        action_mask=old_action_masks)
            
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