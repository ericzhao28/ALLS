"""Selective sampling algorithms for ALSA"""

import random
import math
import numpy as np
import torch
from scipy.stats import mode

from alsa.nets.alt_common import train as sep_train
from alsa.nets.common import train
from alsa.config import LONG_MILESTONES
from alsa.adaptation.label_shift import label_shift


def general_sampling(network,
                     net_cls,
                     dataset,
                     args=None):
    """Query-by-committee (pool): disagreement between random inits of net."""
    # Batch-mode settings
    batch_size = int(dataset.online_len() / args.num_batches)
    label_batch_size = int(args.sample_prop * batch_size)
    labeled_ptrs = np.array([], dtype=np.int32)

    # Initialize committee
    committee = [network]
    if args.sampling_strategy in ["qbc"]:
        for i in range(args.vs_size - 1):
            this_network = net_cls(args.num_cls).to(args.device)
            this_network.train()
            train(this_network,
                  dataset,
                  epochs=args.initial_epochs,
                  args=args,
                  milestones=LONG_MILESTONES)
            this_network.eval()
            committee.append(this_network)

    # Map pointers to label
    if args.diversify == "cheat":
        ys = []
        for i in dataset.indices(split="online"):
            y, _ = dataset.train_labels[i]
            ys.append(y)
        ys = np.array(ys, dtype=np.int32)

    # Begin batch-mode sampling
    for batch_i in range(1, args.num_batches + 1):
        stats = []  # Smaller stats means higher priority
        sep_stats = []  # Bigger value means more important
        with torch.no_grad():
            for image, y, _ in dataset.iterate(batch_size=args.infer_batch_size,
                                               shuffle=False,
                                               split="online"):
                image = image.to(args.device)

                # Aggregate domain sep
                if args.domainsep:
                    this_network = committee[0]
                    this_network.eval()
                    output = torch.exp(this_network(image))
                    p = output.cpu().data.numpy()
                    sep_stats.append(np.sum(-p * np.log(p + 1e-9), axis=1))

                # Produce stats depending on algorithm
                if args.sampling_strategy == "qbc":
                    predictions = []
                    for this_network in committee:
                        this_network.eval()
                        output = torch.exp(this_network(image))
                        p = output.cpu().data.numpy()
                        if not args.train_iw and not args.only_rlls_infer:
                            p = p * dataset.label_weights
                            p = p / np.sum(p, axis=1)[:, None]
                        predictions.append(np.argmax(p, axis=1))
                    predictions = np.stack(predictions)
                    stats.append(mode(predictions, axis=0).count[0])
                if args.sampling_strategy == "bald":
                    predictions = []
                    this_network = committee[0]
                    this_network.train()
                    with torch.no_grad():
                        for _ in range(args.bald_size):
                            output = torch.exp(this_network(image))
                            p = output.cpu().data.numpy()
                            if not args.train_iw and not args.only_rlls_infer:
                                p = p * dataset.label_weights
                                p = p / np.sum(p, axis=1)[:, None]
                            predictions.append(np.argmax(p, axis=1))
                    predictions = np.stack(predictions)
                    stats.append(mode(predictions, axis=0).count[0])
                if args.sampling_strategy == "cheat":
                    this_network = committee[0]
                    this_network.eval()
                    output = torch.exp(this_network(image))
                    p = output.cpu().data.numpy()
                    if not args.train_iw and not args.only_rlls_infer:
                        p = p * dataset.label_weights
                        p = p / np.sum(p, axis=1)[:, None]
                    stats.append(np.equal(np.argmax(p, axis=1), y))
                if args.sampling_strategy == "margin":
                    margin = []
                    this_network = committee[0]
                    this_network.eval()
                    output = torch.exp(this_network(image))
                    p = output.cpu().data.numpy()
                    if not args.train_iw and not args.only_rlls_infer:
                        p = p * dataset.label_weights
                        p = p / np.sum(p, axis=1)[:, None]
                    sorted_p = np.sort(p)
                    margin = sorted_p[:, -1] - sorted_p[:, -2]
                    stats.append(margin)
                if args.sampling_strategy == "maxent":
                    this_network = committee[0]
                    this_network.eval()
                    output = torch.exp(this_network(image))
                    p = output.cpu().data.numpy()
                    if not args.train_iw and not args.only_rlls_infer:
                        p = p * dataset.label_weights
                        p = p / np.sum(p, axis=1)[:, None]
                    stats.append(-np.sum(-p * np.log(p + 1e-9), axis=1))
                if args.sampling_strategy == "random":
                    stats.append(np.random.uniform(size=(len(image),)))

            if args.diversify in ["guess", "overguess"]:
                # Produce new ys
                ys = []
                network.eval()
                for image, _, _ in dataset.iterate(batch_size=args.infer_batch_size,
                                                   shuffle=False,
                                                   split="online"):
                    image = image.to(args.device)
                    output = torch.exp(network(image))
                    p = output.cpu().data.numpy()
                    if not args.train_iw and not args.only_rlls_infer:
                        p = p * dataset.label_weights
                        p = p / np.sum(p, axis=1)[:, None]
                    ys.append(np.argmax(p, axis=1))
                ys = np.concatenate(ys)

        # Concatenate stats
        stats = np.concatenate(stats)

        if args.domainsep:
            sep_stats = np.concatenate(sep_stats)
            sep_stats[sep_stats < 0.5] = 0
            sep_odds = np.uniform(size=sep_stats.shape)
            selection = np.greater(sep_odds, np.uniform(size=sep_stats.shape))
            stats[~selection] = np.infty

        # Stack stats
        new_ptrs = np.setdiff1d(np.arange(len(stats)), labeled_ptrs)
        sorted_ptrs = new_ptrs[np.argsort(stats[new_ptrs])]

        if args.diversify == "none":
            labeled_ptrs = np.concatenate(
                [labeled_ptrs, sorted_ptrs[:label_batch_size]])
        elif args.diversify == "guess":
            # Take top examples from each label
            sorted_ptrs_by_label = {y: [] for y in range(args.num_cls)}
            for ptr in sorted_ptrs:
                sorted_ptrs_by_label[ys[ptr]].append(ptr)

            # Of remaining ptrs per label, find most equal allocation
            label_lens = sorted([len(x)
                                 for x in sorted_ptrs_by_label.values()])
            for i, l in enumerate(label_lens):
                size = math.ceil(
                    (label_batch_size - sum(label_lens[:i])) / len(label_lens[i:]))
                if size <= l:
                    break
                size = -1
            if size == -1:
                raise ValueError()

            # Label pts per each
            for k, ptrs in sorted_ptrs_by_label.items():
                labeled_ptrs = np.concatenate([labeled_ptrs, ptrs[:size]])
            assert len(np.unique(labeled_ptrs)) == len(labeled_ptrs)
        elif args.diversify == "overguess":
            # Take top examples from each label
            sorted_ptrs_by_label = {y: [] for y in range(args.num_cls)}
            for ptr in sorted_ptrs:
                sorted_ptrs_by_label[ys[ptr]].append(ptr)

            # Label pts per each
            for k, ptrs in sorted_ptrs_by_label.items():
                size = math.ceil(
                    dataset.label_weights[k] / sum(dataset.label_weights) * label_batch_size)
                labeled_ptrs = np.concatenate([labeled_ptrs, ptrs[:size]])
            assert len(np.unique(labeled_ptrs)) == len(labeled_ptrs)

        dataset.label_ptrs(labeled_ptrs)

        # Note sample proportion
        print("Sample proportion: ", len(labeled_ptrs) / dataset.online_len())

        # Train networks on current batch status
        if args.domainsep:
            committee[0].train()
            sep_train(committee[0],
                      dataset,
                      epochs=args.partial_epochs,
                      lr=args.finetune_lr,
                      args=args)
            committee[0].eval()

        for this_network in committee:
            this_network.train()
            train(this_network,
                  dataset,
                  epochs=args.partial_epochs,
                  lr=args.finetune_lr,
                  args=args)
            this_network.eval()

        # Handle reweighting procedure
        if args.iterative_iw:
            label_shift(committee[0], dataset, args)

        yield committee[0]


def iwal_bootstrap(network,
                   net_cls,
                   dataset,
                   args=None):
    """IWAL bootstrap instantiation."""
    # Batch-mode settings
    batch_size = int(dataset.online_len() / args.num_batches)
    label_batch_size = int(args.sample_prop * batch_size)

    # Initialize version space
    version_space = [net_cls(args.num_cls).to(args.device)
                     for _ in range(args.vs_size)]
    for this_network in version_space:
        this_network.train()
        train(this_network,
              dataset,
              epochs=args.initial_epochs,
              args=args,
              milestones=LONG_MILESTONES)
        this_network.eval()

    # Process IWAL probabilities
    all_probs = np.zeros(
        (len(version_space), args.num_cls, dataset.online_len()))
    with torch.no_grad():
        for model_i, model in enumerate(version_space):
            model.eval()
            probs = [list() for i in range(args.num_cls)]
            for (data, _, _) in dataset.iterate(batch_size=args.infer_batch_size,
                                                shuffle=False,
                                                split="online"):
                data = data.to(args.device)
                logits = model(data)
                output = torch.exp(logits)  # p(y | x)
                output = output.cpu().data.numpy()
                if not args.train_iw and not args.only_rlls_infer:
                    output = output * dataset.label_weights
                    output = output / np.sum(output, axis=1)[:, None]
                for i in range(args.num_cls):
                    probs[i].append(output[:, i])
            for i, x in enumerate(probs):
                all_probs[model_i, i] = np.concatenate(x)
    all_probs = np.transpose(all_probs, [2, 0, 1])
    # For some datapoint and some label, this is largest disagreement in prob:
    probs_disagreement = np.max(all_probs, axis=2) - np.min(all_probs, axis=2)
    # For some datapoint, this is largest disagreement in prob:
    probs_disagreement = np.max(probs_disagreement, axis=1)
    sample_probs = args.iwal_normalizer + \
        (1 - args.iwal_normalizer) * probs_disagreement
    sample_probs = list(sample_probs)

    # Sample datapoints
    labeled_ptrs = np.array([], dtype=np.int32)
    new_ptrs = []
    for i in range(dataset.online_len()):
        sample_probs[i] = max(sample_probs[i], 0)
        sample_probs[i] = min(sample_probs[i], 1)
        if random.random() < sample_probs[i]:
            new_ptrs.append(i)
    labeled_ptrs = np.concatenate(
        [labeled_ptrs, np.array(new_ptrs, dtype=np.int32)])

    # Train network
    for batch_i in range(1, args.num_batches + 1):
        labeled = label_batch_size * batch_i
        dataset.label_ptrs(labeled_ptrs[:labeled])
        network.train()
        train(network,
              dataset,
              epochs=args.partial_epochs,
              lr=args.finetune_lr,
              args=args)

        print("Sample proportion: ", len(
            labeled_ptrs[:labeled]) / dataset.online_len())
        yield network
