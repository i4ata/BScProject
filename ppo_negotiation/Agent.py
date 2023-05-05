from PPO_negotiation import PPO
from ActivityNet import ActivityNet
from DecisionNet import DecisionNet
from ProposalNet import ProposalNet

import sys
sys.path.append("..")
from rice import Rice

class Agent():
    def __init__(self, env : Rice, intial_state : dict, device : str = 'cpu'):
        self.decisionNet = PPO(model = DecisionNet(env, len(intial_state['features'])), params = None, device = device)
        self.activityNet = PPO(model = ActivityNet(env, len(intial_state['features'])), params = None, device = device)
        self.proposalNet = PPO(model = ProposalNet(env, len(intial_state['features'])), params = None, device = device)