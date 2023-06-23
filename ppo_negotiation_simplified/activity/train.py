from ActingAgent import Agent
import sys
sys.path.append('../..')
from rice_nego_simplfied import Rice

import torch
import numpy as np

from typing import List, Dict

import optuna

def create_envs(n: int) -> List[Rice]:
    return [Rice(i, region_yamls_filename='2_region_yamls') for i in range(n)]

def create_agents(env: Rice, params: dict = None) -> List[Agent]:
    s = env.reset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [Agent(len(s[i]['features']), env.action_space[i], params, device) for i in range(env.num_regions)]

def get_actions(agents: List[Agent], states: List[Dict[str, np.ndarray]]):

    actions = [agents[i].act([state[i] for state in states]) for i in range(len(agents))]
    actions = [{i : actions[i][j] for i in range(len(agents))} for j in range(len(states))]
    return actions

def give_rewards(rewards: Dict[int, float], agents: List[Agent], is_terminal: bool):

    rewards = np.array([list(reward.values()) for reward in rewards])
    for i, agent in enumerate(agents):
        for j in range(len(rewards)):
            agent.ppo.buffer.rewards[j].append(rewards[j, i])
            agent.ppo.buffer.is_terminals[j].append(is_terminal)

def eval_agents(agents: List[Agent], env: Rice, epochs: int) -> float:

    all_rewards = []
    for epoch in range(epochs):
        state = env.reset()
        for t in range(env.episode_length):
            actions = {
                i : agent.eval(state[i], deterministic=False)
                for i, agent in enumerate(agents) 
            }
            state, _, _, _ = env.step(actions)
        all_rewards.append(env.global_state['reward_all_regions']['value'])

    return np.stack(all_rewards).mean((0, 2)).sum()

def objective(trial):

    twos = [2 ** n for n in range(11)]
    params = {
        
        'training': {
            'n_envs' : trial.suggest_int('n_envs', 1, 10),
            'epochs' : trial.suggest_int('epochs', 10, 100),
            'batch_size' : trial.suggest_int('batch_size', 2, 100),
        },

        'ppo': {

            'K_epochs' : trial.suggest_int('K_epochs', 30, 80),
            'gamma' : trial.suggest_float('gamma', .99, .999, log = True),
            'lr_actor' : trial.suggest_float("lr_actor", 1e-4, 1e-1, log=True),
            'lr_critic' : trial.suggest_float("lr_critic", 1e-4, 1e-1, log=True),
            'eps_clip' : trial.suggest_float("eps_clip", 0.1, 0.4),
            'entropy_coef' : trial.suggest_float('entropy_parameter', 0., .2),
            'mse_coef' : trial.suggest_float('critic_loss_parameter', .2, .8),    
        
            'policy': {
                'actor': {
                    'n_hidden_layers_actor' : trial.suggest_int('n_hidden_layers_actor', 1, 6),
                    'hidden_size_actor' : trial.suggest_categorical('hidden_size_actor', twos[3:8]),
                },
                'critic': {
                    'n_hidden_layers_critic' : trial.suggest_int('n_hidden_layers_critic', 1, 6),
                    'hidden_size_critic' : trial.suggest_categorical('hidden_size_critic', twos[3:8]),
                }
            }
        }
    }
    envs = create_envs(params['training']['n_envs'])
    agents = create_agents(envs[0], params['ppo'])
    episode_length = envs[0].episode_length

    # training
    for epoch in range(params['training']['epochs']):
        for batch in range(params['training']['batch_size']):

            states = [env.reset() for env in envs]
            for t in range(episode_length):

                actions = get_actions(agents, states)
                states, rewards, _, _ = zip(*[env.step(actions[i]) for (i, env) in enumerate(envs)])
                give_rewards(rewards, agents, t == episode_length - 1)
                    
        for agent in agents:
            agent.update()

    return eval_agents(agents, envs[0], 100)

if __name__ == '__main__':
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 20, show_progress_bar=True)