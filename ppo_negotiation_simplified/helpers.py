from Agent import Agent
import torch
import numpy as np

import sys
sys.path.append('..')

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

def train(agents: List[Agent], envs: List[Rice], epochs: int = 50, batch_size: int = 40, with_comm = True) -> np.ndarray:

    episode_length = envs[0].episode_length

    eval_stoch = np.zeros((epochs + 1, len(agents)))
    eval_stoch[0] = eval_agents_stoch(agents, envs[0], n_trials=20, with_comm=with_comm)

    eval_det = np.zeros((epochs + 1, len(agents)))
    eval_det[0] = eval_agents_det(agents, envs[0], with_comm=with_comm)

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

        eval_stoch[epoch + 1] = eval_agents_stoch(agents, envs[0], n_trials=20, with_comm=with_comm)
        eval_det[epoch + 1] = eval_agents_det(agents, envs[0], with_comm=with_comm)
        
    return eval_stoch, eval_det

def random_runs(env: Rice, n_trials: int = 20, with_comm = False):

    return eval_agents_stoch(create_agents(env), env, n_trials=n_trials, with_comm=with_comm)

def eval_agents_stoch(agents: List[Agent], env: Rice, n_trials = 20, with_comm = True):
    env_rewards = np.zeros((n_trials, env.episode_length, len(agents))) 
    for trial in range(n_trials):
        state = env.reset()
        for step in range(env.episode_length):

            if with_comm:
                proposals = np.concatenate([agent.eval_make_proposals(state[i]) for i, agent in enumerate(agents)])
                state = env.register_proposals(proposals)

                decisions = np.concatenate([agent.eval_make_decisions(state[i]) for i, agent in enumerate(agents)])
                state = env.register_decisions(decisions)

            actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}
            state, reward, _, _ = env.step(actions)
            env_rewards[trial, step] = list(reward.values())
    return env_rewards.mean((0, 1))

def eval_agents_det(agents: List[Agent], env: Rice, with_comm = True) -> np.ndarray:
    env_rewards = np.zeros((env.episode_length, len(agents)))
    state = env.reset()
    for step in range(env.episode_length):
        
        if with_comm:
            proposals = np.concatenate([agent.eval_make_proposals(state[i], True) for i, agent in enumerate(agents)])
            state = env.register_proposals(proposals)

            decisions = np.concatenate([agent.eval_make_decisions(state[i], True) for i, agent in enumerate(agents)])
            state = env.register_decisions(decisions)

        actions = {i: agent.eval_act(state[i], deterministic=True) for i, agent in enumerate(agents)}
        state, reward, _, _ = env.step(actions)
        env_rewards[step] = list(reward.values())

    return env_rewards.mean(0)


def run_experiments(n_agents):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    envs = create_envs(yamls_filename = f'yamls/{n_agents}_region_yamls')
    assert envs[0].num_regions == n_agents

    agents_no_comm = create_agents(envs[0])
    agents_with_comm = create_agents(envs[0])

    stoch_no_comm, det_no_comm = train(agents_no_comm, envs, epochs=200, batch_size=50, with_comm=False)
    stoch_with_comm, det_with_comm = train(agents_with_comm, envs, epochs=200, batch_size=50, with_comm=True)
    
    np.save(f'runs/{n_agents}/stoch_no_comm.npy', stoch_no_comm)
    np.save(f'runs/{n_agents}/det_no_comm.npy', det_no_comm)
    np.save(f'runs/{n_agents}/stoch_with_comm.npy', stoch_with_comm)
    np.save(f'runs/{n_agents}det_with_comm.npy', det_with_comm)