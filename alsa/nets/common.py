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


def get_preds(
        model,
        dataset_iter,
        device,
        args,  # pylint: disable=W0613
        top_k=None,
        label_weights=None,
        rlls_infer=True):
    '''Perform model inference'''
    preds = []
    top_k_preds = []
    true_preds = []
    for (data, labels, _) in dataset_iter:
        data = data.to(device)
        output = model(data)
        if label_weights is not None:
            p = torch.exp(output).cpu().data.numpy()
            if rlls_infer:
                p = p * label_weights
            preds.append(np.argmax(p, axis=1))
        else:
            preds.append(torch.argmax(output, dim=1).cpu().data.numpy())
        if top_k is not None:
            top_k_preds.append(
                output.topk(top_k, dim=1).indices.cpu().data.numpy())
        true_preds.append(labels.cpu().data.numpy())
    preds = np.concatenate(preds)
    true_preds = np.concatenate(true_preds)
    if top_k is None:
        return preds, true_preds
    top_k_preds = np.concatenate(top_k_preds)
    return preds, true_preds, top_k_preds


def evaluate(model, dataset_iter, device, args, label_weights=None):
    '''Evaluate on test dataset'''
    model.eval()
    with torch.no_grad():
        preds, true_preds, top5_preds = get_preds(
            model,
            dataset_iter,
            device,
            args,
            top_k=5,
            label_weights=None if args.train_iw else label_weights,
            rlls_infer=args.rlls_infer)
    total_acc = np.sum(preds == true_preds) / len(preds)
    total_top5_acc = np.sum(
        np.any(np.repeat(np.expand_dims(true_preds, 1), 5, axis=1)
               == top5_preds,
               axis=1)) / len(preds)
    accuracies = {"Accuracy": total_acc, "Top-5 Accuracy": total_top5_acc}
    cls_avg = []
    for i in np.unique(true_preds):
        correct_cls = np.sum(np.logical_and(preds == i, true_preds == i))
        total_cls = np.sum(true_preds == i)
        accuracies["Class %s Accuracy" % i] = correct_cls / total_cls
        cls_avg.append(correct_cls / total_cls)
    accuracies["Average class accuracy"] = np.mean(cls_avg)
    accuracies["Macro F1"] = metrics.f1_score(true_preds,
                                              preds,
                                              average='macro')
    accuracies["Micro F1"] = metrics.f1_score(true_preds,
                                              preds,
                                              average='micro')
    accuracies["Weighted F1"] = metrics.f1_score(true_preds,
                                                 preds,
                                                 average='weighted')
    return accuracies


def train(model, dataset, epochs, args, lr=None, log_fn=None, milestones=None):
    '''Traing model'''
    model.train().to(args.device)
    lr = lr or args.lr
    if milestones is None:
        if args.partial_epochs <= 10:
            milestones = SHORT_MILESTONES
        elif args.partial_epochs <= 30:
            milestones = MID_MILESTONES

    if args.dataset == "mnist":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=milestones,
                                                   gamma=args.gamma)
        warmup_scheduler = WarmUpLR(
            optimizer,
            math.ceil(dataset.labeled_len() / args.batch_size) *
            args.warm_epochs)

    for epoch in range(epochs):
        print("Epoch", epoch)
        for (data, target,
             weight) in dataset.iterate(batch_size=args.batch_size,
                                        shuffle=True,
                                        split="labeled"):
            if args.dataset == "cifar100" or args.dataset == "cifar" or args.dataset == "nabirds":
                if epoch <= args.warm_epochs:
                    warmup_scheduler.step()

            data, target, weight = data.to(args.device), target.to(
                args.device), weight.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            if args.train_iw:
                loss = F.nll_loss(output, target, reduction="none")
                loss = loss * weight  # importance weighting
                loss = torch.mean(loss)
            else:
                loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        if args.dataset == "mnist" or epoch > args.warm_epochs:
            scheduler.step()

        if log_fn is not None:
            if epoch % 2 == 1:
                model.eval()
                log_fn(epoch)
                model.train()


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]
