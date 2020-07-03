"""Architectures for CIFAR dataset"""

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from alsa.nets.resnet import resnet18


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
        """Forward  pass for domain separator logic"""
        assert self.training == self._net.training
        x = self._net.forward_sep(x)
        output = F.log_softmax(x, dim=1)
        return output
