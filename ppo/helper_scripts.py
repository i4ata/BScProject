import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from copy import deepcopy

from PPO import PPO

import sys
sys.path.insert(0, '..')
from rice import Rice

def create_agents(env : Rice, params : dict, device : str) -> List[PPO]:
    """
    Create agents that receive only the features of the environment as inputs, (not the action masks)
    """
    agents = []
    initial_state = env.reset()
    for i in range(env.num_regions):
        agents.append(
            PPO(
                state_dim = len(initial_state[i]['features']), 
                action_dim = env.action_space[i],
                params = params,
                device = device
            )
        )
    return agents

def evaluate_agents(agents : List[PPO], env : Rice) -> Tuple[dict, dict, dict]:
    """
    Evaluate agents on a new environment episode (policies are deterministic)
    """
    state = env.reset()
    actions = {i : [] for i in range(len(agents))}
    rewards = {i : [] for i in range(len(agents))}

    for _ in range(env.episode_length):
        collective_action = {}
        for agent_id in range(len(agents)):
            #action = agents[agent_id].select_action(state[agent_id])
            action = agents[agent_id].select_action_deterministically(state[agent_id])
            collective_action[agent_id] = action
            actions[agent_id].append(action)
        state, reward, _, _ = env.step(collective_action)
        
        for agent_id in range(len(agents)):
            rewards[agent_id].append(reward[agent_id])

    return deepcopy(env.global_state), actions, rewards

def baseline(env : Rice) -> Tuple[dict, dict, dict]:
    """
    Evaluate random agents on a new environment episode
    """
    return evaluate_agents(create_agents(env), env)

def plot_losses(agents : List[PPO], updating_epochs : int) -> None:
    fig, axs = plt.subplots(len(agents), sharex=True, figsize = (10, 10))
    fig.suptitle('Losses over time')
    for idx, agent in enumerate(agents):
        axs[idx].plot(agent.loss_collection, label = f"Agent {idx}")
        axs[idx].set_ylabel("Loss Value")
        axs[idx].legend()
        #axs[idx].grid()
        axs[idx].axhline(c = "red")

        for update in range(len(agents[0].loss_collection))[::updating_epochs]:
            axs[idx].axvline(update, c = 'orange', alpha = .5, ls = '--')
    plt.xlabel("Training episode")

    plt.show()

def plot_rewards(rewards : Dict[int, List[float]]) -> None:
    fig, axs = plt.subplots(len(rewards), sharex=True, figsize = (10, 10))
    fig.suptitle('Rewards over time')
    for idx in rewards:
        axs[idx].plot(rewards[idx], alpha = .2)
        axs[idx].plot(np.convolve(rewards[idx], np.ones(100)/100, mode='valid'), 
                      label = f'Agent {idx}', c = 'red')
        axs[idx].set_ylabel("Reward")
        axs[idx].legend()
        axs[idx].grid()
    plt.xlabel("Timestep")
    plt.show()
