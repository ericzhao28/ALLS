"""Architectures for CIFAR dataset"""

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from alsa.nets.resnet import resnet18
from alsa.nets.densenet import DenseNet_Cifar


def ccCIFARNet(num_classes, depth=100, k=12):
    """Build DensetNet"""
    N = (depth - 4) // 6
    model = DenseNet_Cifar(growth_rate=k,
                           block_config=[N, N, N],
                           num_init_features=2*k,
                           num_classes=num_classes)
    return model


class CIFARNet(nn.Module):
    '''Resnet18 + log-softmax for p.'''

    def __init__(self, output_dim):
        super(CIFARNet, self).__init__()
        self._net = resnet18(num_classes=output_dim)
        super(CIFARNet, self).add_module('resnet', self._net)

    def forward(self, x):  # pylint: disable=W0221
        assert self.training == self._net.training
        x = self._net(x)
        output = F.log_softmax(x, dim=1)
        return output

    def forward_sep(self, x):  # pylint: disable=W0221
        assert self.training == self._net.training
        x = self._net.forward_sep(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCIFARNet(nn.Module):
    '''Simple CNN from CIFAR baselines.'''

    def __init__(self, output_dim):
        super(SimpleCIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):  # pylint: disable=W0221
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
