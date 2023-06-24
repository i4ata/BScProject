from Agent import Agent
import torch
import numpy as np
from rice_nego_simplfied import Rice

from tqdm import tqdm

from typing import List, Dict, Tuple, Optional

def create_envs(n: int = 5, 
                yamls_filename: str = '2_region_yamls', 
                num_discrete_action_levels: int = 10) -> List[Rice]:
    
    return [Rice(i, num_discrete_action_levels, yamls_filename) for i in range(n)]

def create_agents(env: Rice) -> List[Agent]:
    state = env.reset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [Agent(len(state[i]['features']), env.action_space[i], env.num_regions, i, device) 
            for i in range(env.num_regions)]

def get_mean_reward(env: Rice, n_trials = 100) -> np.ndarray:
    return env.estimate_reward_distribution(n_trials)

def proposals_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    
    # Get actions in the shape [agent_id, {'proposals' : [env_id, proposal], 'promise' : [env_id, promise]}]
    actions = [agent.make_proposals([state[i] for state in states]) for i, agent in enumerate(agents)]
        
    # Transform actions to [env_id, {agent_id, : {'proposals' : proposal, 'promise', 'promise'}}]
    actions = [
        [
            {'proposals' : actions[i]['proposals'][j], 'promises' : actions[i]['promises'][j]}
            for i in range(len(agents))
        ]
        for j in range(len(envs))
    ]
    
    # Pass the proposals to the environments
    states = [env.register_proposals(actions[i]) for i, env in enumerate(envs)]
    return states

def decisions_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:

    # Get actions in the shape [agent_id, env_id, decision]
    actions = [agent.make_decisions([state[agent.id] for state in states]) for agent in agents]
    
    # Transform actions to [env_id, {agent_id : decision}]
    actions = [
        [actions[agent_id][env_id] for agent_id in range(len(agents))]
        for env_id in range(len(envs))
    ]
    
    # Pass the decisions to the environments
    states = [env.register_decisions(actions[i]) for i, env in enumerate(envs)]
    return states

def action_step(agents: List[Agent], 
                envs: List[Rice], 
                states: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    
    actions = [agents[i].act([state[i] for state in states]) for i in range(len(agents))]
    actions = [{i : actions[i][j] for i in range(len(agents))} for j in range(len(envs))]
    states, _, _, _ = zip(*[env.step(actions[i]) for (i, env) in enumerate(envs)])
    
    return states

def give_rewards(agents: List[Agent], 
                 envs: List[Rice], 
                 communication_on: bool = True,
                 mean_rewards: Optional[np.ndarray] = None) -> None:

    rewards = np.stack([env.global_state['reward_all_regions']['value'][1:] for env in envs])
    is_terminals = np.zeros(envs[0].episode_length, dtype=bool)
    is_terminals[-1] = True
    
    if communication_on:
        mean_action_rewards = rewards.mean(2)
        nego_rewards = np.stack([env.global_negotiation_state['rewards'][:-1] for env in envs])
        nego_rewards *= mean_rewards
        nego_rewards += mean_action_rewards[:, :, np.newaxis]

    for i, agent in enumerate(agents):
        for j in range(len(envs)):
            agent.activity_net.buffer.rewards[j].extend(rewards[j, :, i])
            agent.activity_net.buffer.is_terminals[j].extend(is_terminals)

            if communication_on:
                agent.proposal_net.buffer.rewards[j].extend(nego_rewards[j, :, i])
                agent.decision_net.buffer.rewards[j].extend(nego_rewards[j, :, i])

                agent.proposal_net.buffer.is_terminals[j].extend(is_terminals)
                agent.decision_net.buffer.is_terminals[j].extend(is_terminals)


def train(agents: List[Agent], 
          envs: List[Rice], 
          epochs: int = 70, 
          batch_size: int = 64, 
          communication_on: bool = True, 
          mean_rewards: Optional[np.ndarray] = None) -> np.ndarray:
    
    episode_length = envs[0].episode_length
    eval_rewards = np.zeros((epochs + 1, len(agents)))
    eval_rewards[0] = eval_agents(agents, envs[0], communication_on)
    for epoch in tqdm(range(epochs)):
        
        for batch in range(batch_size):
            states = [env.reset() for env in envs]
            for t in range(episode_length):
                
                if communication_on:
                    states = proposals_step(agents, envs, states)
                    states = decisions_step(agents, envs, states)
                    
                states = action_step(agents, envs, states)
                
            give_rewards(agents, envs, communication_on, mean_rewards)

        for agent in agents:
            agent.update(communication_on)

        eval_rewards[epoch + 1] = eval_agents(agents, envs[0], communication_on)

    return eval_rewards
        

def random_runs(env: Rice, n_trials: int = 10):
    ret = []
    for trial in range(n_trials):
        env.random_run()
        ret.append(env.global_state)
    return ret

def eval_agents(agents: List[Agent], env: Rice, communication_on: bool = True, n_trials: int = 20) -> np.ndarray:

    env_rewards = np.zeros((n_trials, env.episode_length, len(agents))) 

    for trial in range(n_trials):

        state = env.reset()

        for step in range(env.episode_length):

            if communication_on:
                proposals = [agent.eval_make_proposals(state[i]) for i, agent in enumerate(agents)]
                state = env.register_proposals(proposals)
                    
                decisions = [agent.eval_make_decisions(state[i]) for i, agent in enumerate(agents)]
                state = env.register_decisions(decisions)

            # actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}
            actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}

            state, reward, _, _ = env.step(actions)

            env_rewards[trial, step] = list(reward.values())

    return env_rewards.mean((0, 1))