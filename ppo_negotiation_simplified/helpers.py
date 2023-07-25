from Agent import Agent
import torch
import numpy as np

import sys
import os
sys.path.append('..')

from rice_nego_simplfied import Rice

from tqdm import tqdm

from typing import List, Dict, Tuple

gamma = .99
episode_length = 20
discount_factors = gamma ** np.arange(episode_length)

def create_envs(n: int = 5, 
                yamls_filename: str = 'yamls/2_region_yamls', 
                num_discrete_action_levels: int = 10) -> List[Rice]:
    
    return [Rice(i, num_discrete_action_levels, yamls_filename) for i in range(n)]

def create_agents(env: Rice) -> List[Agent]:
    state = env.reset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [Agent(len(state[i]['features']), env.action_space[i], env.num_regions, i, device) 
            for i in range(env.num_regions)]

def get_baseline_cum_returns(env: Rice, n_trials = 100) -> np.ndarray:
    returns = np.zeros(n_trials)
    for trial in tqdm(range(n_trials)):

        env.reset()
        for step in range(env.episode_length):
            env.step({
                i : np.random.randint(env.num_discrete_action_levels, size=len(env.action_space[i]))
                for i in range(env.num_regions)
            })

        returns[trial] = np.mean(compute_returns(env.global_state['reward_all_regions']['value']))

    return returns   

def proposals_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[List[Dict[str, np.ndarray]]]) -> List[Dict[str, np.ndarray]]:
    
    actions = np.stack([agent.make_proposals([s[i] for s in states]) for i, agent in enumerate(agents)], axis = 1)
    new_states = [env.register_proposals(action) for action, env in zip(actions.astype(bool), envs)]
    return new_states

def decisions_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[List[Dict[str, np.ndarray]]]) -> List[Dict[str, np.ndarray]]:

    actions = np.stack([agent.make_decisions([s[i] for s in states]) for i, agent in enumerate(agents)], axis = 1)
    new_states = [env.register_decisions(action) for action, env in zip(actions.astype(bool), envs)]
    return new_states

def action_step(agents: List[Agent], 
                envs: List[Rice], 
                states: List[List[Dict[str, np.ndarray]]]) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, float]]]:
    
    actions = np.stack([agents[i].act([state[i] for state in states]) for i in range(len(agents))], axis = 1)
    states, rewards, _, _ = zip(*[env.step(dict(enumerate(action))) for action, env in zip(actions, envs)])
    
    return states, rewards

def train(agents: List[Agent], 
          envs: List[Rice], 
          epochs: int = 50, 
          batch_size: int = 40, 
          with_comm: bool = True,
          eval_epochs: int = 50) -> np.ndarray:

    episode_length = envs[0].episode_length

    evaluation = np.zeros((epochs + 1, eval_epochs, len(agents)))
    evaluation[0] = eval_agents(agents, envs[0], n_trials=eval_epochs, with_comm=with_comm)

    for epoch in tqdm(range(epochs)):
        
        for batch in range(batch_size):
            states = [env.reset() for env in envs]
            for t in range(episode_length):
                    
                if with_comm:
                    states = proposals_step(agents, envs, states)
                    states = decisions_step(agents, envs, states)

                states, rewards = action_step(agents, envs, states)

                rewards = np.array([list(r.values()) for r in rewards])
                mean_rewards = rewards.mean(1)
                is_terminal = t == episode_length - 1

                for i, agent in enumerate(agents):
                    for env_id in range(len(envs)):
                        
                        if with_comm:

                            agent.proposal_net.buffer.rewards[env_id].append(mean_rewards[env_id])
                            agent.proposal_net.buffer.is_terminals[env_id].append(is_terminal)

                            agent.decision_net.buffer.rewards[env_id].append(mean_rewards[env_id])
                            agent.decision_net.buffer.is_terminals[env_id].append(is_terminal)

                        agent.activity_net.buffer.rewards[env_id].append(rewards[env_id, i])
                        agent.activity_net.buffer.is_terminals[env_id].append(is_terminal)
        
        for agent in agents:

            agent.activity_net.update()

            if with_comm:
                agent.decision_net.update()
                agent.proposal_net.update()

        evaluation[epoch + 1] = eval_agents(agents, envs[0], n_trials=eval_epochs, with_comm=with_comm)
        
    return evaluation

def compute_returns(rewards: np.ndarray) -> np.ndarray:
    cum_returns = np.cumsum(np.flipud(rewards[1:]), axis = 0)
    disc_cum_returns = cum_returns * discount_factors[:, np.newaxis]
    return np.sum(disc_cum_returns, 0)

def eval_agents(agents: List[Agent], env: Rice, n_trials = 50, with_comm = True) -> np.ndarray:
    env_rewards = np.zeros((n_trials, len(agents))) 
    for trial in range(n_trials):
        state = env.reset()
        for step in range(env.episode_length):

            if with_comm:
                proposals = np.concatenate([agent.eval_make_proposals(state[i]) for i, agent in enumerate(agents)])
                state = env.register_proposals(proposals)

                decisions = np.concatenate([agent.eval_make_decisions(state[i]) for i, agent in enumerate(agents)])
                state = env.register_decisions(decisions)

            actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}
            state, _, _, _ = env.step(actions)
        env_rewards[trial] = compute_returns(env.global_state['reward_all_regions']['value'])
    return env_rewards

def run_experiments(n_agents, epochs = 200, batch_size = 50):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    envs = create_envs(yamls_filename = f'yamls/{n_agents}_region_yamls')
    assert envs[0].num_regions == n_agents

    agents_no_comm = create_agents(envs[0])
    agents_with_comm = create_agents(envs[0])

    path = f'runs/{n_agents}/no_comm'
    os.makedirs(path, exist_ok=True)
    stoch_no_comm = train(agents_no_comm, envs, epochs=epochs, batch_size=batch_size, with_comm=False)
    for agent in agents_no_comm:
        agent.save(path, False)
    np.save(path + '/stoch.npy', stoch_no_comm)

    path = f'runs/{n_agents}/with_comm'
    os.makedirs(path, exist_ok=True)
    stoch_with_comm = train(agents_with_comm, envs, epochs=epochs, batch_size=batch_size, with_comm=True)
    for agent in agents_with_comm:
        agent.save(path, True)
    np.save(path + '/stoch.npy', stoch_with_comm)

def eval_agents_final(agents: List[Agent], env: Rice, n_trials = 1000, with_comm = True):
    
    rewards, decisions, promises, proposals = np.zeros((4, n_trials))
    for trial in tqdm(range(n_trials)):
        state = env.reset()
        for step in range(env.episode_length):

            if with_comm:
                p = np.concatenate([agent.eval_make_proposals(state[i]) for i, agent in enumerate(agents)])
                state = env.register_proposals(p)

                d = np.concatenate([agent.eval_make_decisions(state[i]) for i, agent in enumerate(agents)])
                state = env.register_decisions(d)

            actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}
            state, _, _, _ = env.step(actions)

        rewards[trial] = compute_returns(env.global_state['reward_all_regions']['value']).mean()
        if with_comm:
            decisions[trial] = env.global_negotiation_state['decisions'][:-1].mean()
            proposals[trial] = env.global_negotiation_state['proposals'][:-1].mean()
            promises[trial] = env.global_negotiation_state['promises'][:-1].mean()
    
    if with_comm:
        return rewards, decisions, promises, proposals
    return rewards

def run_experiment_2(epochs = 200, batch_size = 50):
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    envs = create_envs(yamls_filename=f'yamls/2_region_yamls')
    agents1 = create_agents(envs[0])
    agents2 = create_agents(envs[0])
    
    train(agents1, envs, epochs = epochs, batch_size=batch_size, eval_epochs=1)
    train(agents2, envs, epochs = epochs, batch_size=batch_size, eval_epochs=1)

    rs, decs, proms, props = eval_agents_final((agents1[0], agents2[0]), envs[0])
    
    path = 'runs2'
    os.makedirs(path, exist_ok=True)
    np.save(path + 'rs.npy', rs)
    np.save(path + 'decs.npy', decs)
    np.save(path + 'proms.npy', proms)
    np.save(path + 'props.npy', props)