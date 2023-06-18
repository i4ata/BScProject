import torch
from Interfaces import PPO, ActorCritic

from typing import Dict, List, Tuple
import numpy as np

################################## PPO Policy ##################################

class PPONegotiation(PPO):
    def __init__(self, model: ActorCritic, params: dict, device: str):
        super().__init__(model, params, device)
    

    def select_action(self, env_state, **kwargs) -> Dict[str, List[np.ndarray]]:
        """
        Select action in state `state` and fill in the rollout buffer with the relevant data
        """

        actions, state_val = self.policy_old.act(env_state, **kwargs)

        return_dict = {
            'decisions' : [decision.cpu().numpy() for decision in actions['decisions']['decisions']],
            'proposals' : [proposal.cpu().numpy() for proposal in actions['proposals']['proposals']],
            'promises'  : [promise. cpu().numpy() for promise  in actions['promises' ]['promises' ]],

            # 'messages' : [message.cpu().numpy() for message in actions['messages']]
        }

        if kwargs['save_state']:

            self.buffer.    env_states.                     extend(env_state)
            #self.buffer.    proposals_states.               extend(kwargs['proposals'])
            #self.buffer.    promises_states.                extend(kwargs['promises'])

        if kwargs['save_decisions']:

            self.buffer.    decisions.                      extend(actions['decisions']['decisions'])
            self.buffer.    decisions_logprobs.             extend(actions['decisions']['log_probs'])

            self.buffer.    hidden_states_decisions_actor.  extend(zip(*self.policy_old.actor.hidden_state))
            self.buffer.    hidden_states_decisions_critic. extend(zip(*self.policy_old.critic.hidden_state))

            self.buffer.    state_values_decisions.         extend(state_val)

            # self.buffer.messages_decisions.extend(actions['messages'])
            # self.buffer.messages_state_proposals.extend(kwargs['messages'])

            self.buffer.proposals_states_2.extend(kwargs['proposals'])
            self.buffer.promises_states_2.extend(kwargs['promises'])


        if kwargs['save_proposals_promises']:

            self.buffer.    proposals.                      extend(actions['proposals']['proposals'])
            self.buffer.    proposals_logprobs.             extend(actions['proposals']['log_probs'])

            self.buffer.    promises.                       extend(actions['promises']['promises'])
            self.buffer.    promises_logprobs.              extend(actions['promises']['log_probs'])

            self.buffer.    hidden_states_proposals_actor.  extend(zip(*self.policy_old.actor.hidden_state))
            self.buffer.    hidden_states_proposals_critic. extend(zip(*self.policy_old.critic.hidden_state))

            self.buffer.    state_values_proposals.         extend(state_val)

            # self.buffer.messages_proposals.extend(actions['messages'])
            # self.buffer.messages_state_decisions.extend(kwargs['messages'])

            self.buffer.proposals_states.extend(kwargs['proposals'])
            self.buffer.promises_states.extend(kwargs['promises'])
            

        return return_dict
    
    def update(self) -> None:
        """
        Update the policy with PPO
        """

        # Monte Carlo estimate of returns
        returns = []
        discounted_return = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards),
            reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + .9 * discounted_return
            returns.insert(0, discounted_return)

        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # convert list to tensor
        old_states                          = torch.stack(self.buffer.env_states)                       .detach()
        old_decisions                       = torch.stack(self.buffer.decisions)                        .detach()
        old_decisions_logprobs              = torch.stack(self.buffer.decisions_logprobs)               .detach()
        old_proposals                       = torch.stack(self.buffer.proposals)                        .detach()
        old_proposals_logprobs              = torch.stack(self.buffer.proposals_logprobs)               .detach()
        old_promises                        = torch.stack(self.buffer.promises)                         .detach()
        old_promises_logprobs               = torch.stack(self.buffer.promises_logprobs)                .detach()
        old_proposals_states                = torch.stack(self.buffer.proposals_states)                 .detach()
        old_promises_states                 = torch.stack(self.buffer.promises_states)                  .detach()
        
        old_state_values_decisions          = torch.stack(self.buffer.state_values_decisions)           .detach()
        old_state_values_proposals          = torch.stack(self.buffer.state_values_proposals)           .detach()

        old_hidden_states_decisions_actor   = self.to_tuple(self.buffer.hidden_states_decisions_actor)
        old_hidden_states_decisions_critic  = self.to_tuple(self.buffer.hidden_states_decisions_critic)
        
        old_hidden_states_proposals_actor   = self.to_tuple(self.buffer.hidden_states_proposals_actor)
        old_hidden_states_proposals_critic  = self.to_tuple(self.buffer.hidden_states_decisions_critic)
        
        # old_messages_states_proposals = torch.stack(self.buffer.messages_state_proposals).detach()
        # old_messages_states_decisions = torch.stack(self.buffer.messages_state_decisions).detach()

        # old_messages_proposals = torch.stack(self.buffer.messages_proposals).detach()
        # old_messages_decisions = torch.stack(self.buffer.messages_decisions).detach()

        old_promises_states_2 = torch.stack(self.buffer.promises_states_2).detach()
        old_proposals_states_2 = torch.stack(self.buffer.proposals_states_2).detach()

        # calculate advantages
        advantages_decisions = (returns.detach() - old_state_values_decisions.squeeze()).unsqueeze(-1).unsqueeze(-1)
        advantages_proposals = (returns.detach() - old_state_values_proposals.squeeze()).unsqueeze(-1).unsqueeze(-1)
                    

        # Optimize policy for K epochs
        for _ in range(10):
            # Evaluating old actions and values
            evaluation = self.policy.evaluate(
                old_states, 
                actions = {
                    'decisions': old_decisions,
                    'proposals': old_proposals,
                    'promises' : old_promises,
                    # 'messages' : {
                    #     'decisions' : old_messages_decisions,
                    #     'proposals' : old_messages_proposals
                    # }
                },
                states = {
                    'proposals': old_proposals_states,
                    'promises' : old_promises_states,

                    'proposals2': old_proposals_states_2,
                    'promises2' : old_promises_states_2 
                },
                hidden_states = {
                    'decisions' : {
                        'actor' : old_hidden_states_decisions_actor,
                        'critic' : old_hidden_states_decisions_critic,
                    },
                    'proposals' : {
                        'actor' : old_hidden_states_proposals_actor,
                        'critic' : old_hidden_states_proposals_critic
                    }
                },
                # messages = {
                #     'decisions' : old_messages_states_decisions,
                #     'proposals' : old_messages_states_proposals
                # }
            )
            actions = {k : evaluation[k] for k in ['decisions', 'promises', 'proposals']}
            state_values_decisions = torch.squeeze(evaluation['state_values']['decision'])
            state_values_proposals = torch.squeeze(evaluation['state_values']['proposal'])

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios_decisions = torch.exp(actions['decisions']['log_probs'] - old_decisions_logprobs.detach())
            ratios_proposals = torch.exp(actions['proposals']['log_probs'] - old_proposals_logprobs.detach())
            ratios_promises  = torch.exp(actions['promises' ]['log_probs'] - old_promises_logprobs .detach())
            
            # Finding Surrogate Loss
            surr1_decisions = ratios_decisions * advantages_decisions
            surr2_decisions = torch.clamp(ratios_decisions, .8, 1.2) * advantages_decisions

            surr1_proposals = ratios_proposals * advantages_proposals
            surr2_proposals = torch.clamp(ratios_proposals, .8, 1.2) * advantages_proposals

            surr1_promises  = ratios_promises  * advantages_proposals
            surr2_promises  = torch.clamp(ratios_promises,  .8, 1.2) * advantages_proposals

            # final loss of clipped objective PPO
            loss = -torch.min(surr1_proposals, surr2_proposals) - \
                    torch.min(surr1_decisions, surr2_decisions) - \
                    torch.min(surr1_promises,  surr2_promises) + \
                    .5 * self.MseLoss(state_values_proposals, returns) + \
                    .5 * self.MseLoss(state_values_decisions, returns) - \
                    .01 * actions['decisions']['entropies'] - \
                    .01 * actions['proposals']['entropies'] - \
                    .01 * actions['promises' ]['entropies']
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

    # Transform a list of 1d hidden states to 1 2d hidden state
    def to_tuple(self, xs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.stack([x[0] for x in xs]).detach(), torch.stack([x[1] for x in xs]).detach())
