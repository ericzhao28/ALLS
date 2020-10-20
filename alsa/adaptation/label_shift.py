"""Label shift algorithms: BBSE, RLLS, uniform, cheat."""

import numpy as np
import cvxpy as cp
import torch

from alsa.nets.common import get_preds


def label_shift(network, dataset, args):
    """Obtain label shift weights"""

    # Disable dropout in networks.
    network.eval()

    # Find priors and confusion for source split.
    source_dataset = dataset.iterate(
        args.infer_batch_size,
        shuffle=False,
        split="labeled" if args.reweight else "warmstart")
    with torch.no_grad():
        preds, true_preds = get_preds(network,
                                      source_dataset,
                                      args.device,
                                      args,
                                      label_weights=dataset.label_weights)
    source_priors = np.zeros(args.num_cls)
    for i in range(args.num_cls):
        source_priors[i] = float(len(np.where(preds == i)[0])) / len(preds)
    true_source_priors = np.zeros(args.num_cls)
    for i in range(args.num_cls):
        true_source_priors[i] = float(len(
            np.where(true_preds == i)[0])) / len(true_preds)
    confuse = np.zeros((args.num_cls, args.num_cls))
    for i in range(args.num_cls):
        for j in range(args.num_cls):
            idxs = np.where((preds == i) & (true_preds == j))[0]
            confuse[i, j] = float(len(idxs)) / len(preds)
    print(confuse)

    # Find priors for target split.
    target_dataset = dataset.iterate(args.infer_batch_size,
                                     shuffle=False,
                                     split="test")
    with torch.no_grad():
        preds, true_preds = get_preds(network,
                                      target_dataset,
                                      args.device,
                                      args,
                                      label_weights=dataset.label_weights)
    true_target_priors = np.zeros(args.num_cls)
    for i in range(args.num_cls):
        true_target_priors[i] = float(len(
            np.where(true_preds == i)[0])) / len(true_preds)
    target_priors = np.zeros(args.num_cls)
    for i in range(args.num_cls):
        target_priors[i] = float(len(np.where(preds == i)[0])) / len(preds)

    # Report true weights
    true_label_weights = true_target_priors / true_source_priors
    print("Optimal label weights:",
          np.array(100 * true_label_weights, dtype=np.int32))

    # w = q(y) / p(y)
    if args.shift_correction == "cheat":
        label_weights = (1 - args.rlls_lambda) + \
            true_label_weights * args.rlls_lambda
    if args.shift_correction == "none":
        label_weights = np.ones(args.num_cls, dtype=np.float32)
    if args.shift_correction == "rlls":
        # Optimize eq (6) as convex optm problem
        theta = cp.Variable(args.num_cls)
        b = target_priors - source_priors
        objective = cp.Minimize(
            cp.pnorm(confuse @ theta - b) + args.rlls_reg * cp.pnorm(theta))
        constraints = [-1 <= theta]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        label_weights = 1 + args.rlls_lambda * theta.value
    if args.shift_correction == "bbse":
        # Compute good guess (BBSE)
        try:
            label_weights = np.linalg.inv(confuse) @ target_priors
        except np.linalg.LinAlgError:
            label_weights = np.linalg.inv(confuse + np.random.uniform(
                0, 1e-6, size=confuse.shape)) @ target_priors
        label_weights[label_weights < 0] = 0

    print("Label weights:", np.array(100 * label_weights, dtype=np.int32))

    diff_weights = label_weights - true_label_weights
    print("Diff weights:", np.array(100 * diff_weights, dtype=np.int32))
    print("Diff weights l2 {}, mse {}, max {}".format(
        np.linalg.norm(diff_weights), np.mean(np.square(diff_weights)),
        np.amax(np.abs(diff_weights))))

    # Precomputed per-example weights
    dataset.set_weight(label_weights)

    if dataset.first_weight is None:
        dataset.first_weight = true_label_weights

    return np.mean(np.square(diff_weights))
