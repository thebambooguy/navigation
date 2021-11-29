from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """DQN Agent"""

    def __init__(self, state_size, action_size, seed):
        """
        Initialize parameters and build model.
        :param int state_size: Dimension of each state
        :param int action_size: Dimension of each action
        :param int seed: Random seed
        """

        super(QNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        torch.manual_seed(seed)

        self.network = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.state_size, 64)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(64, 64)),
            ('relu2', nn.ReLU()),
            ('fc4', nn.Linear(64, self.action_size)),
        ]))

    def forward(self, state):
        """
        Forward pass of the network.
        :param int state: Current state of the agent
        :return Output of the network
        """
        return self.network(state)
