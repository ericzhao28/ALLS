"""Architectures for MNIST dataset"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    '''Simple CNN from MNIST baselines.'''
    def __init__(self, output_dim=10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):  # pylint: disable=W0221
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleMNISTNet(nn.Module):
    '''Simple CNN from MNIST baselines.'''
    def __init__(self, output_dim=10):
        super(SimpleMNISTNet, self).__init__()
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):  # pylint: disable=W0221
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
