"""Common torch network utilities for training/testing"""

from __future__ import print_function

import math
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
from alsa.config import SHORT_MILESTONES, MID_MILESTONES


def train(model, dataset, epochs, args, lr=None, log_fn=None, milestones=None):
    '''Traing model'''
    model.train().to(args.device)
    lr = lr or args.lr
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)

    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model._net.alt_fc.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        print("Epoch", epoch)
        for (data, target) in dataset.domain_iterate(batch_size=args.batch_size, shuffle=True):
            data, target, weight = data.to(args.device), target.to(
                args.device), weight.to(args.device)
            optimizer.zero_grad()
            output = model.forward_sep(data)
            if args.train_iw:
                loss = F.nll_loss(output, target, reduction="none")
                loss = loss * weight  # importance weighting
                loss = torch.mean(loss)
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    for param in model.parameters():
        param.requires_grad = True
