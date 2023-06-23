from Agent import Agent

import torch

import numpy as np
import pandas as pd

import os
import sys
os.chdir("..")
from rice_nego_simplfied import Rice

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy

from typing import List, Dict, Tuple, Optional

def create_envs(n: int = 5, 
                yamls_filename: str = '2_region_yamls', 
                num_discrete_action_levels: int = 10) -> List[Rice]:
    
    return [Rice(i, num_discrete_action_levels, yamls_filename) for i in range(n)]

def create_agents(env: Rice) -> List[Agent]:
    state = env.reset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [Agent(len(state[i]['features']), env.action_space[i], env.num_regions, i, device) for i in range(env.num_regions)]

def get_mean_reward(env: Rice, n_trials = 100) -> np.ndarray:
    return env.estimate_reward_distribution(n_trials)

def proposals_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    
    # Get actions in the shape [agent_id, {'proposals' : [env_id, proposal], 'promise' : [env_id, promise]}]
    actions = [agent.make_proposals([state[agent.id] for state in states]) for agent in agents]
        
    # Transform actions to [env_id, {agent_id, : {'proposals' : proposal, 'promise', 'promise'}}]
    actions = [
        [
            {
                'proposals' : actions[agent_id]['proposals'][env_id],
                'promises' : actions[agent_id]['promises'][env_id]
            }
            for agent_id in range(len(agents))
        ]
        for env_id in range(len(envs))
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
        [
            actions[agent_id][env_id]
            for agent_id in range(len(agents))
        ]
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
          epochs: int = 70, 
          batch_size: int = 64, 
          communication_on: bool = True, 
          mean_rewards: Optional[np.ndarray] = None) -> np.ndarray:
    
    episode_length = envs[0].episode_length
    eval_rewards = np.zeros((epochs, len(agents)))
    for epoch in tqdm(range(epochs)):
        
        for batch in range(batch_size):
            states = [env.reset() for env in envs]
            for t in range(episode_length):
                
                terminals = [t == episode_length - 1] * len(envs)
                if communication_on:
                    states = proposals_step(agents, envs, states)
                    states = decisions_step(agents, envs, states)
                    
                states, rewards = action_step(agents, envs, states)
                
                rewards = np.array([list(reward.values()) for reward in rewards])
                mean_action_rewards = rewards.mean(1)
                
                for agent in agents:

                    agent.nets['activityNet'].buffer.rewards.extend(rewards[:, agent.id])
                    agent.nets['activityNet'].buffer.is_terminals.extend(terminals)
                    
                    if not communication_on:
                        continue

                    nego_rewards = np.array([env.global_negotiation_state['rewards'][t][agent.id] for env in envs])
                    print(nego_rewards)
                    
                    nego_rewards *= mean_rewards[agent.id]
                    print(nego_rewards)
                    nego_rewards += mean_action_rewards
                    print(nego_rewards)
                    agent.nets['proposalNet'].buffer.rewards.extend(nego_rewards)
                    agent.nets['decisionNet'].buffer.rewards.extend(nego_rewards)

                    agent.nets['proposalNet'].buffer.is_terminals.extend(terminals)
                    agent.nets['decisionNet'].buffer.is_terminals.extend(terminals)
                b   
        for agent in agents:
            agent.update(communication_on)

        eval_rewards[epoch] = eval_agents(agents, envs[0], communication_on)

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
            actions = {i: agent.eval(state[i]) for i, agent in enumerate(agents)}

            state, reward, _, _ = env.step(actions)

            env_rewards[trial, step] = list(reward.values())

    return env_rewards.mean((0, 1))