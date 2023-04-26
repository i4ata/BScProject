from PPO import PPO
from helper_scripts import create_agents
import sys
sys.path.append("..")
from rice import Rice

import numpy as np

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def objective(trial):
    twos = [2 ** n for n in range(11)]
    params = {
        'n_envs' : trial.suggest_int('n_envs', 1, 10),
        'epochs' : trial.suggest_int('epochs', 10, 100),
        'batch_size' : trial.suggest_categorical('batch_size', twos[:5]),
        'K_epochs' : trial.suggest_int('K_epochs', 30, 80),

        'n_layers_actor' : trial.suggest_int('n_layers_actor', 1, 10),
        'n_layers_critic' : trial.suggest_int('n_layers_critic', 1, 10),
        'hidden_size_actor' : trial.suggest_categorical('hidden_size_actor', twos[4:9]),
        'hidden_size_critic' : trial.suggest_categorical('hidden_size_critic', twos[4:9]),
        
        'gamma' : trial.suggest_float('gamma', .99, .999, log = True),
        'lr_actor' : trial.suggest_float("lr_actor", 1e-3, 1e-1, log=True),
        'lr_critic' : trial.suggest_float("lr_critic", 1e-3, 1e-1, log=True),
        'eps_clip' : trial.suggest_float("eps_clip", 0.1, 0.4),
        'entropy_parameter' : trial.suggest_float('entropy_parameter', 0., .1),
        'critic_loss_parameter' : trial.suggest_float('critic_loss_parameter', .2, .8)
    }

    envs = [Rice(region_yamls_filename='fewer_region_yamls') for _ in range(params['n_envs'])]
    agents = create_agents(envs[0], params, device)
    episode_length = envs[0].episode_length

    for epoch in range(params['epochs']):
        for batch in range(params['batch_size']):
            states = [env.reset() for env in envs]
            for t in range(episode_length):
                collective_action = {}
                for agent_id in range(len(agents)):
                    collective_action[agent_id] = agents[agent_id].select_action(
                        [state[agent_id] for state in states]
                    )
                    
                states, rewards, _, _ = zip(*[
                    env.step({agent_id : collective_action[agent_id][i] 
                            for agent_id in range(len(agents))}) 
                    for (i, env) in enumerate(envs)
                ])
                
                for agent_id in range(len(agents)):
                    r = [reward[agent_id] for reward in rewards]
                    agents[agent_id].buffer.rewards.extend(r)    
                    agents[agent_id].buffer.is_terminals.extend([t == episode_length - 1] * len(envs))

        for agent in agents:
            agent.update()
        #rewards_collection.append(eval(agents, envs[0]))

    return eval_stochastic(agents, envs[0])

def eval_stochastic(agents, env, evaluation_steps = 100):
    mean_reward = 0
    for step in range(evaluation_steps):
        state = env.reset()
        for t in range(env.episode_length):
            collective_action = {
                i : agents[i].select_action_stochastically(state[i])
                for i in range(len(agents))
            }
            state, reward, _, _ = env.step(collective_action)
            mean_reward += sum(reward.values())
    return mean_reward / (env.episode_length * len(agents) * evaluation_steps)

def eval(agents, env):
    rewards = {i : [] for i in range(len(agents))}
    state = env.reset()
    for _ in range(env.episode_length):
        collective_action = {}
        for agent_id in range(len(agents)):
            action = agents[agent_id].select_action_deterministically(state[agent_id])
            collective_action[agent_id] = action
        state, reward, _, _ = env.step(collective_action)
        
        for agent_id in range(len(agents)):
            rewards[agent_id].append(reward[agent_id])
    
    return np.mean(list(rewards.values()))