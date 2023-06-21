from Agent import Agent

import torch

import numpy as np
import pandas as pd

import os
import sys
os.chdir("..")
from rice_nego import Rice

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy

from typing import List, Dict, Tuple, Optional

def create_envs(n: int = 5, 
                yamls_filename: str = '2_region_yamls', 
                num_discrete_action_levels: int = 10, 
                nego_steps: int = 5) -> List[Rice]:
    
    return [Rice(i, num_discrete_action_levels, yamls_filename, nego_steps) for i in range(n)]

def create_agents(env: Rice) -> List[Agent]:
    state = env.reset()
    return [Agent(len(state[i]['features']), env.action_space[i], env.num_regions, i) for i in range(env.num_regions)]

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
          epochs: int = 100, 
          batch_size: int = 5, 
          communication_on: bool = True, 
          mean_rewards: Optional[np.ndarray] = None):
    
    episode_length = envs[0].episode_length
    negotiation_steps = envs[0].max_negotiation_steps

    # The terminal states are the same every time
    is_terminals_negotiation = np.zeros(len(envs) * negotiation_steps, dtype = bool)
    is_terminals_negotiation[negotiation_steps - 1 :: negotiation_steps] = True
    is_terminals_negotiation = np.repeat(is_terminals_negotiation, episode_length)

    all_proposal_rewards = [[] for agent in agents]
    all_decision_rewards = [[] for agent in agents]
    all_action_rewards = [[] for agent in agents]

    for epoch in tqdm(range(epochs)):
        
        for batch in range(batch_size):
            states = [env.reset() for env in envs]
            for t in range(episode_length):
                
                if communication_on:
                    for agent in agents:
                        agent.reset_negotiation_hs()

                    for step in range(negotiation_steps):
                        states = proposals_step(agents, envs, states)
                        states = decisions_step(agents, envs, states)
                    
                states, rewards = action_step(agents, envs, states)
                
                rewards = np.array([list(reward.values()) for reward in rewards])
                mean_action_rewards = rewards.mean(1).reshape(1, -1) * (len(agents) - 1)

                for agent in agents:

                    agent.nets['activityNet'].buffer.rewards.extend(rewards[:, agent.id])
                    agent.nets['activityNet'].buffer.is_terminals.extend([t == episode_length - 1] * len(envs))
                    all_action_rewards[agent.id].extend(rewards[:, agent.id])

                    if not communication_on:
                        continue

                    # Get proposal and decision rewards from envs, make them [nego_step, env_id], scale them
                    # to match the order of magnitude of the environment rewards
                    proposal_rewards, decision_rewards = [np.stack([
                        [env.global_negotiation_state[rs][t][step][agent.id] for env in envs]
                        for step in range(negotiation_steps)]
                    ) * mean_rewards[agent.id] for rs in ('rewards_proposals', 'rewards_decisions')]

                    proposal_rewards_added, decision_rewards_added = \
                        [np.where(rs != 0, rs + mean_action_rewards, rs).flatten()
                        for rs in (proposal_rewards, decision_rewards)]

                    agent.nets['proposalNet'].buffer.rewards.extend(proposal_rewards_added)
                    agent.nets['decisionNet'].buffer.rewards.extend(decision_rewards_added)

                    agent.nets['proposalNet'].buffer.is_terminals.extend(is_terminals_negotiation)
                    agent.nets['decisionNet'].buffer.is_terminals.extend(is_terminals_negotiation)

                    all_proposal_rewards[agent.id].extend(proposal_rewards_added[proposal_rewards_added != 0])
                    all_decision_rewards[agent.id].extend(decision_rewards_added[decision_rewards_added != 0])

                
        for agent in agents:
            agent.update(communication_on)

    if not communication_on:
        return all_action_rewards
    return {'action' : all_action_rewards, 'decision' : all_decision_rewards, 'proposal' : all_proposal_rewards}

def random_runs(env: Rice, n_trials: int = 10):
    ret = []
    for trial in range(n_trials):
        env.random_run()
        ret.append(env.global_state)
    return ret

def eval_agents(agents: List[Agent], env: Rice, communication_on: bool = True, n_trials: int = 10):

    ret = {'global_states' : [], 'nego_states' : []} if communication_on else []

    for trial in range(n_trials):

        state = env.reset()

        for step in range(env.episode_length):

            if communication_on:
                for agent in agents:
                    agent.reset_negotiation_hs()

                for step in range(env.max_negotiation_steps):
                
                    proposals = [agent.eval_make_proposals(state[agent.id]) for agent in agents]
                    state = env.register_proposals(proposals)
                    
                    decisions = [agent.eval_make_decisions(state[agent.id]) for agent in agents]
                    state = env.register_decisions(decisions)

            actions = {agent.id: agent.eval_act(state[agent.id]) for agent in agents}

            state, _, _, _ = env.step(actions)

        if communication_on:
            ret['global_states'].append(copy.deepcopy(env.global_state))
            ret['nego_states'].append(copy.deepcopy(env.global_negotiation_state))
        else:
            ret.append(copy.deepcopy(env.global_state))

    return ret