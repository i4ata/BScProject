from gym.spaces import MultiDiscrete, Dict, Box
from torch.distributions import Categorical
import numpy as np
import torch


from action_net import ActionNet

_BIG_NUMBER = 1e20

class Agent:

    def __init__(self, 
                 observation_space : Dict, 
                 action_space : MultiDiscrete, 
                 id : int):
        
        self.id = id
        self.nets = [ActionNet(observation_space, action_space)] # will put the negotiation networks here

    def act(self,
            stage : int, 
            observation : Dict):
        
        probs = self.nets[stage](observation)

        actions, log_probs = zip(*[self.get_action_and_log_prob(space) for space in probs])
        log_probs = torch.cat([log_prob.unsqueeze(0) for log_prob in log_probs]) 
        return actions, log_probs
    
    def get_action_and_log_prob(self, space):
        d = Categorical(logits = space)
        action = d.sample()
        return action.item(), d.log_prob(action)
    
# Taken from the original repo    
def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations

    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment

    Returns:
        spaces.Dict: A dictionary of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for key, val in obs.items():

        # list of lists are 'listified' np arrays
        _val = val
        if isinstance(val, list):
            _val = np.array(val)
        elif isinstance(val, (int, np.integer, float, np.floating)):
            _val = np.array([val])

        # assign Space
        if isinstance(_val, np.ndarray):
            large_num = float(_BIG_NUMBER)
            box = Box(
                low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
            )
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                large_num = large_num // 2
                box = Box(
                    low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[key] = box

        elif isinstance(_val, dict):
            dict_of_spaces[key] = recursive_obs_dict_to_spaces_dict(_val)
        else:
            raise TypeError
    return Dict(dict_of_spaces)