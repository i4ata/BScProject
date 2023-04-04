from agent import Agent, recursive_obs_dict_to_spaces_dict
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
from rice import Rice

def evaluate_agents(env : Rice, agents : List[Agent]) -> Tuple[dict, list, list]:
    """
    Measure performance on trained agents on a brand new environment without learning.
    Returns: (global state of the environment, list of collective actions, list of rewards)
    """
    state = env.reset()
    collective_actions = [[] for _ in agents]
    all_rewards = [[] for _ in agents]
    for i in range(env.episode_length):
        collective_action = {}
        for agent in agents:
            action, _ = agent.act(0, state[agent.id])
            collective_action[agent.id] = np.array(action)
            collective_actions[agent.id].append(action)
        state, reward, _, _ = env.step(collective_action)
        
        for agent in agents:
            all_rewards[agent.id].append(reward[agent.id])

    return env.global_state, collective_actions, all_rewards

def baseline(env : Rice) -> Tuple[dict, list, list]:
    """
    Measure performance on trained agents on a brand new environment without learning.
    Returns: (global state of the environment, list of collective actions, list of rewards)
    """
    return evaluate_agents(env, create_agents(env))

def create_agents(env : Rice) -> List[Agent]:
    """
    Create agents suitable for the environment.
    """
    initial_state = env.reset()
    agents = []
    for agent_id in initial_state:
        agents.append(
            Agent(
                #observation_space = get_observation_space(agent_id),
                #action_space = get_action_space(agent_id),
                observation_space = recursive_obs_dict_to_spaces_dict(initial_state[agent_id]),
                action_space = env.action_space[agent_id],
                id = agent_id
            )
        )
    return agents

def plot_losses(losses : List[List[float]]) -> None:
    fig, axs = plt.subplots(len(losses), sharex=True, figsize = (10, 10))
    fig.suptitle('Losses over time')
    for idx, loss in enumerate(losses):
        axs[idx].plot(loss, label = f"Agent {idx}")
        axs[idx].set_ylabel("Loss Value")
    plt.xlabel("Training episode")
    plt.legend()
    plt.show()
        