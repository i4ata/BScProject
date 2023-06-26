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
    actions = [agent.make_decisions([state[i] for state in states]) for i, agent in enumerate(agents)]
    
    # Transform actions to [env_id, {agent_id : decision}]
    actions = [
        np.concatenate([actions[agent_id][env_id] for agent_id in range(len(agents))])
        for env_id in range(len(envs))
    ]
    
    # Pass the decisions to the environments
    states = [env.register_decisions(actions[i]) for i, env in enumerate(envs)]
    return states

def action_step(agents: List[Agent], 
                envs: List[Rice], 
                states: List[Dict[str, np.ndarray]]) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, float]]]:
    
    actions = [
        agents[agent_id].act([state[agent_id] for state in states]) 
        for agent_id in range(len(agents))
    ]
    
    actions = [
        {
            agent_id : actions[agent_id][env_id]
            for agent_id in range(len(agents))
        }
        for env_id in range(len(envs))
    ]
    
    states, rewards, _, _ = zip(*[env.step(actions[i]) for (i, env) in enumerate(envs)])
    
    return states, rewards

def train(agents: List[Agent], 
          envs: List[Rice], 
          epochs: int = 20, 
          batch_size: int = 20) -> np.ndarray:
    
    episode_length = envs[0].episode_length
    eval_rewards = np.zeros((epochs + 1, len(agents)))
    eval_rewards[0] = eval_agents(agents, envs[0])

    eval_rewards_deterministic = np.zeros((epochs + 1, len(agents)))
    eval_rewards_deterministic[0] = eval_agents(agents, envs[0])

    for epoch in tqdm(range(epochs)):
        
        for batch in range(batch_size):
            states = [env.reset() for env in envs]
            for t in range(episode_length):
                    
                states, rewards = action_step(agents, envs, states)
                
                rewards = np.array([list(reward.values()) for reward in rewards])
                
                for i, agent in enumerate(agents):
                    for env_id in range(len(envs)):
                        agent.activity_net.buffer.rewards[env_id].append(rewards[env_id, i])
                        agent.activity_net.buffer.is_terminals[env_id].append(t == episode_length - 1)
                    
        for agent in agents:
            agent.update(False)

        eval_rewards[epoch + 1] = eval_agents(agents, envs[0])
        eval_rewards_deterministic[epoch + 1] = eval_deterministic(agents, envs[0])

    return eval_rewards, eval_rewards_deterministic
        
def random_runs(env: Rice, n_trials: int = 20):

    return eval_agents(create_agents(env), env, communication_on=False, n_trials=n_trials)

def eval_agents(agents: List[Agent], env: Rice, n_trials: int = 20) -> np.ndarray:
    env_rewards = np.zeros((n_trials, env.episode_length, len(agents))) 
    for trial in range(n_trials):
        state = env.reset()
        for step in range(env.episode_length):
            actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}
            state, reward, _, _ = env.step(actions)
            env_rewards[trial, step] = list(reward.values())
    return env_rewards.mean((0, 1))

def eval_deterministic(agents: List[Agent], env: Rice) -> np.ndarray:
    env_rewards = np.zeros((env.episode_length, len(agents)))
    state = env.reset()
    for step in range(env.episode_length):
        actions = {i: agent.eval_act(state[i], deterministic=True) for i, agent in enumerate(agents)}
        state, reward, _, _ = env.step(actions)
        env_rewards[step] = list(reward.values())
    return env_rewards.mean(0)