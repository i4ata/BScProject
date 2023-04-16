from gym.spaces import MultiDiscrete
from torch.distributions import Categorical
import torch


from action_net import ActionNet

class Agent:

    def __init__(self, 
                 observation_space : int, 
                 action_space : MultiDiscrete, 
                 id : int):
        
        self.id = id
        self.nets = [ActionNet(observation_space, action_space)] # will put the negotiation networks here

    def act(self,
            stage : int, 
            observation : dict):
        
        probs = self.nets[stage](observation)
        actions, log_probs = zip(*[self.get_action_and_log_prob(space) for space in probs])
        log_probs = torch.cat([log_prob.unsqueeze(0) for log_prob in log_probs]) 
        return actions, log_probs
    
    def get_action_and_log_prob(self, space):
        d = Categorical(logits = space)
        action = d.sample()
        return action.item(), d.log_prob(action)