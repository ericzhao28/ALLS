# -*- coding: utf-8 -*-

import random
from comet_ml import Experiment
import torch
import numpy as np
from alsa.config import comet_ml_key
from alsa.main.args import get_args, get_net_cls
from alsa.datasets.datasets import get_datasets
from alsa.nets.common import evaluate, train


def experiment():
    """Baseline exp"""
    args = get_args(None)
    logger = Experiment(
        comet_ml_key,
        project_name="active-label-shift-adaptation")
    logger.set_name("Baseline lr {} g {} bs {} {} {}".format(
        args.lr,
        args.gamma,
        args.batch_size,
        "simple " if args.simple_model else "",
        args.dataset
    ))
    logger.log_parameters(vars(args))

    # Seed the experiment
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("Running seed ", seed)
    torch.cuda.set_device(args.device)
    assert torch.cuda.is_available()

    # Shuffle dataset
    dataset = get_datasets(args)

    # Train h0
    net_cls = get_net_cls(args)
    network = net_cls(args.num_cls).to(args.device)

    def log_fn(epoch):
        network.eval()
        accuracy = evaluate(network, dataset.iterate(
            args.infer_batch_size, False, split="test"), args.device, args)
        logger.log_metrics(accuracy, step=epoch)
    train(
        network,
        dataset=dataset,
        epochs=args.initial_epochs,
        args=args,
        log_fn=log_fn)


if __name__ == "__main__":
    experiment()
