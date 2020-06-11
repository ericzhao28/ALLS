"""Architectures for BIRD dataset"""

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from alsa.nets.resnet import resnet50, resnet18, resnet34


class BIRDNet(nn.Module):
    '''Resnet50 + log-softmax for p.'''

    def __init__(self, output_dim):
        super(BIRDNet, self).__init__()
        self._net = resnet34(num_classes=output_dim)
        # self._net = resnet50(num_classes=output_dim)
        super(BIRDNet, self).add_module('resnet', self._net)

    def forward(self, x):  # pylint: disable=W0221
        assert self.training == self._net.training
        x = self._net(x)
        output = F.log_softmax(x, dim=1)
        return output
