import train

env = train.create_envs(2)[0]
agents = train.create_agents(env)
train.eval_agents(agents, env, 20)