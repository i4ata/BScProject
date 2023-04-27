import torch.nn as nn

class DecisionNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(DecisionNet, self).__init__()

        # action space needs to be equal to the number of agents. The output of the model is the probability
        # of accepting each proposal
        # state space needs to be the number of agents * 2 * size of action mask
        self.layers = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.layers(x)

    def _flatten_and_reshape(self, x):
        return x.reshape(x.shape[0], -1)