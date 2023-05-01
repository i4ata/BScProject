from PPO_negotiation import PPO
from ActivityNet import ActivityNet
from DecisionNet import DecisionNet

import sys
sys.path.append("..")
from rice import Rice

class Agent():
    def __init__(self, env : Rice, intial_state : dict):
        self.nets = [DecisionNet(env, len(intial_state['features'])), ActivityNet(env, len(intial_state['features']))]