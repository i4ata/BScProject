import torch
import numpy as np

from activity.PPOActivity import PPOActivity
from gym.spaces import MultiDiscrete

from typing import Dict, List

class Agent():

    def __init__(self, state_space: int, action_space: MultiDiscrete, params: dict = None, device: str = 'cpu') -> None:
        
        self.device = device
        self.ppo = PPOActivity(state_space, action_space, params, device)

    def act(self, states: List[Dict[str, np.ndarray]]) -> List[np.ndarray]:
        
        features = torch.FloatTensor(np.stack([
            state['features'] for state in states
        ])).to(self.device)

        masks = torch.FloatTensor(np.stack([
            state['action_mask'].flatten() for state in states
        ])).to(self.device)

        return self.ppo.select_action(features, masks)

    def update(self):
        self.ppo.update()

    def eval_act(self, state: Dict[str, np.ndarray], deterministic = False) -> np.ndarray:

        features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        mask = torch.FloatTensor(state['action_mask'].flatten()).unsqueeze(0).to(self.device)

        actions = self.ppo.policy.eval_act(features, mask, deterministic)
        return actions