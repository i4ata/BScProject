from Agent import Agent
import torch
import numpy as np
from rice_nego import Rice

from tqdm import tqdm

from typing import List, Dict, Tuple, Optional

def create_envs(n: int = 5, 
                yamls_filename: str = '2_region_yamls', 
                num_discrete_action_levels: int = 10) -> List[Rice]:
    
    return [Rice(i, num_discrete_action_levels, yamls_filename) for i in range(n)]

def create_agents(env: Rice) -> List[Agent]:
    state = env.reset()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return [Agent(len(state[i]['features']), env.action_space[i], i, device) 
            for i in range(env.num_regions)]

def get_mean_reward(env: Rice, n_trials = 100) -> np.ndarray:
    return env.estimate_reward_distribution(n_trials)

def proposals_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[List[Dict[str, np.ndarray]]]) -> List[Dict[str, np.ndarray]]:
    
    pps, pms = 'proposals', 'promises'
    actions = [agent.make_proposals([state[i] for state in states]) for i, agent in enumerate(agents)]
    actions = [[{pps : actions[i][pps][j], pms : actions[i][pms][j]} for i in range(len(agents))] for j in range(len(envs))]
    states = [env.register_proposals(actions[i]) for i, env in enumerate(envs)]

    return states

def decisions_step(agents: List[Agent], 
                   envs: List[Rice], 
                   states: List[List[Dict[str, np.ndarray]]]) -> List[Dict[str, np.ndarray]]:

    actions = [agent.make_decisions([state[i] for state in states]) for i, agent in enumerate(agents)]
    actions = [np.concatenate([actions[i][j] for i in range(len(agents))]) for j in range(len(envs))]
    states = [env.register_decisions(actions[i]) for i, env in enumerate(envs)]
    return states

def action_step(agents: List[Agent], 
                envs: List[Rice], 
                states: List[List[Dict[str, np.ndarray]]]) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, float]]]:
    
    actions = [agents[i].act([state[i] for state in states]) for i in range(len(agents))]
    actions = [{i : actions[i][j] for i in range(len(agents))} for j in range(len(envs))]
    states, rewards, _, _ = zip(*[env.step(actions[i]) for (i, env) in enumerate(envs)])
    
    return states, rewards

def train_no_comm(agents: List[Agent], envs: List[Rice], epochs: int = 50, batch_size: int = 40) -> np.ndarray:
    
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
            agent.activity_net.update()

        eval_rewards[epoch + 1] = eval_agents(agents, envs[0])
        eval_rewards_deterministic[epoch + 1] = eval_deterministic(agents, envs[0])

    return eval_rewards, eval_rewards_deterministic

def train_with_comm(agents: List[Agent], envs: List[Rice], epochs: int = 50, batch_size: int = 40) -> np.ndarray:

    episode_length = envs[0].episode_length
    negotiation_length = envs[0].max_negotiation_steps
    
    eval_rewards = np.zeros((epochs + 1, len(agents)))
    eval_rewards[0] = eval_agents(agents, envs[0])

    eval_rewards_deterministic = np.zeros((epochs + 1, len(agents)))
    eval_rewards_deterministic[0] = eval_agents(agents, envs[0])

    # print('Estimating mean reward ...')
    # mean_reward = get_mean_reward(env=envs[0]).mean()

    for epoch in tqdm(range(epochs)):
        
        for batch in range(batch_size):
            states = [env.reset() for env in envs]

            for agent in agents:
                agent.reset_hs()

            for t in range(episode_length):

                for nego_step in range(negotiation_length):
                        
                    states = proposals_step(agents, envs, states)
                    states = decisions_step(agents, envs, states)

                #states, rewards_actions = action_step_eval(agents, envs, states)
                states, rewards_actions = action_step(agents, envs, states)

                rewards_actions = np.array([list(reward.values()) for reward in rewards_actions])
                mean_rewards_action = rewards_actions.mean(1)
                # rewards_negotiation = np.array(rewards_negotiation)
                
                # # rewards_negotiation = rewards_negotiation * mean_reward / 2 + rewards_actions.mean(1)
                # rewards_negotiation = rewards_negotiation * mean_reward / 3 + rewards_actions.mean(1)
                is_terminal = t == episode_length - 1

                is_terminal_negotiation = np.zeros(negotiation_length, dtype=bool)
                is_terminal_negotiation[-1] = True
                nego_rewards = np.zeros((len(envs), negotiation_length))
                for i, env in enumerate(envs):
                    idx = np.argmax(env.global_negotiation_state['negotiation_status'][t])
                    nego_rewards[i, idx] = mean_rewards_action[i]

                for i, agent in enumerate(agents):
                    for env_id in range(len(envs)):
                        # agent.proposal_net.buffer.rewards[env_id].append(rewards_negotiation[env_id])
                        agent.proposal_net.buffer.rewards[env_id].extend(nego_rewards[env_id])
                        agent.proposal_net.buffer.is_terminals[env_id].extend(is_terminal_negotiation)

                        agent.decision_net.buffer.rewards[env_id].extend(nego_rewards[env_id])
                        agent.decision_net.buffer.is_terminals[env_id].extend(is_terminal_negotiation)

                        agent.activity_net.buffer.rewards[env_id].append(rewards_actions[env_id, i])
                        agent.activity_net.buffer.is_terminals[env_id].append(is_terminal)

        for agent in agents:
            agent.activity_net.update()
            agent.decision_net.update()
            agent.proposal_net.update()

        eval_rewards[epoch + 1] = eval_agents_with_comm(agents, envs[0])
        eval_rewards_deterministic[epoch + 1] = eval_agents_with_comm_deterministic(agents, envs[0])

    return eval_rewards, eval_rewards_deterministic

def random_runs(env: Rice, n_trials: int = 20):

    return eval_agents(create_agents(env), env, communication_on=False, n_trials=n_trials)

def action_step_eval(agents: List[Agent], 
                     envs: List[Rice], 
                     states: List[List[Dict[str, np.ndarray]]]) -> Tuple[List[List[Dict[str, np.ndarray]]], List[float]]:
    new_states, rewards, _, _ = \
        zip(*[env.step({i : agent.eval_act(states[j][i]) for i, agent in enumerate(agents)}) for j, env in enumerate(envs)])
    return new_states, rewards

def eval_agents_with_comm_deterministic(agents: List[Agent], env: Rice) -> np.ndarray:
    env_rewards = np.zeros((env.episode_length, len(agents)))
    state = env.reset()

    for agent in agents:
        agent.reset_hs()

    for step in range(env.episode_length):

        for nego_step in range(env.max_negotiation_steps):
            proposals = [agent.eval_make_proposals(state[i], deterministic=True) for i, agent in enumerate(agents)]
            state = env.register_proposals(proposals)

            decisions = [agent.eval_make_decisions(state[i], deterministic=True) for i, agent in enumerate(agents)]
            state = env.register_decisions(decisions)

        actions = {i: agent.eval_act(state[i], deterministic=True) for i, agent in enumerate(agents)}
        state, reward, _, _ = env.step(actions)
        env_rewards[step] = list(reward.values())
    return env_rewards.mean(0)

def eval_agents_with_comm(agents: List[Agent], env: Rice, n_trials = 20) -> np.ndarray:
    env_rewards = np.zeros((n_trials, env.episode_length, len(agents))) 
    for trial in range(n_trials):
        state = env.reset()

        for agent in agents:
            agent.reset_hs()

        for step in range(env.episode_length):

            for nego_step in range(env.max_negotiation_steps):
                proposals = [agent.eval_make_proposals(state[i]) for i, agent in enumerate(agents)]
                state = env.register_proposals(proposals)

                decisions = [agent.eval_make_decisions(state[i]) for i, agent in enumerate(agents)]
                state = env.register_decisions(decisions)

            actions = {i: agent.eval_act(state[i]) for i, agent in enumerate(agents)}
            state, reward, _, _ = env.step(actions)
            env_rewards[trial, step] = list(reward.values())
    return env_rewards.mean((0, 1))

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