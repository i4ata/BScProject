from ActingAgent import Agent
import sys
sys.path.append('../..')
from rice_nego import Rice

import torch
import numpy as np

import optuna

def objective(trial):

    twos = [2 ** n for n in range(11)]
    params = {
        
        'training': {
            'n_envs' : trial.suggest_int('n_envs', 1, 10),
            'epochs' : trial.suggest_int('epochs', 10, 100),
            'batch_size' : trial.suggest_categorical('batch_size', twos[3:7]),
        },

        'ppo': {

            'K_epochs' : trial.suggest_int('K_epochs', 30, 80),
            'gamma' : trial.suggest_float('gamma', .99, .999, log = True),
            'lr_actor' : trial.suggest_float("lr_actor", 1e-3, 1e-1, log=True),
            'lr_critic' : trial.suggest_float("lr_critic", 1e-3, 1e-1, log=True),
            'eps_clip' : trial.suggest_float("eps_clip", 0.1, 0.4),
            'entropy_coef' : trial.suggest_float('entropy_parameter', 0., .1),
            'mse_coef' : trial.suggest_float('critic_loss_parameter', .2, .8),    
        
            'policy': {
                'actor': {
                    'n_hidden_layers_actor' : trial.suggest_int('n_hidden_layers_actor', 1, 10),
                    'hidden_size_actor' : trial.suggest_categorical('hidden_size_actor', twos[4:9]),
                },
                'critic': {
                    'n_hidden_layers_critic' : trial.suggest_int('n_hidden_layers_critic', 1, 10),
                    'hidden_size_critic' : trial.suggest_categorical('hidden_size_critic', twos[4:9]),
                }
            }
        }
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    envs = [Rice(i, region_yamls_filename='2_region_yamls') 
            for i in range(params['training']['n_envs'])]
    initial_state = envs[0].reset()
    agents = [Agent(len(initial_state[0]['features']), envs[0].action_space[0], params['ppo'], device) 
              for i in range(envs[0].num_regions)]
    episode_length = envs[0].episode_length

    # training
    for epoch in range(params['training']['epochs']):
        for batch in range(params['training']['batch_size']):
            states = [env.reset() for env in envs]
            for t in range(episode_length):

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
                rewards = np.array([list(reward.values()) for reward in rewards])

                for i, agent in enumerate(agents):
                    agent.ppo.buffer.rewards.extend(rewards[:, i])
                    agent.ppo.buffer.is_terminals.extend([t == episode_length - 1] * len(envs))
                    
        for agent in agents:
            agent.update()

    # evaluate
    eval_epochs = 20
    all_rewards = []
    eval_env = envs[0]
    for epoch in range(eval_epochs):
        state = eval_env.reset()
        for t in episode_length:
            actions = {
                agent_id : agent.eval(state[0], deterministic=False)
                for agent_id, agent in enumerate(agents) 
            }
            eval_env.step(actions)
        all_rewards.append(np.array(eval_env.global_state['rewards_all_regions']))

    return np.stack(all_rewards).mean(0, 1).sum()

if __name__ == '__main__':
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 5)