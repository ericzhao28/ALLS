"""Adaptation utilities (largely for label shift)."""

import numpy as np


def measure_composition(dataset):
    """Find ideal label weights and compute shift."""

    # Count true label distributions in test and labeled sets
    target_data_by_label = {k: 0 for k in dataset.label_space}
    source_data_by_label = {k: 0 for k in dataset.label_space}
    for i in dataset.indices(split="labeled"):
        y, _ = dataset.train_labels[i]
        source_data_by_label[y] += 1
    for i in dataset.indices(split="test"):
        y, _ = dataset.test_labels[i]
        target_data_by_label[y] += 1

    # Compute normalized prior vectors, observing common traversal of labels
    keys = sorted(target_data_by_label.keys())

    target_priors = [target_data_by_label[k] for k in keys]
    target_priors = np.array(target_priors, dtype=np.float32)
    target_priors /= np.sum(target_priors)

    source_priors = [source_data_by_label[k] for k in keys]
    source_priors = np.array(source_priors, dtype=np.float32)
    source_priors /= np.sum(source_priors)

    # Find theta given by q(y) / p(y) - 1
    label_weights = target_priors / source_priors
    label_shift = np.linalg.norm(label_weights - 1)

    # Find theta assuming q(y) = 1
    uniform_label_weights = 1.0 / source_priors
    uniform_label_shift = np.linalg.norm(uniform_label_weights - 1)

    return label_weights, label_shift, uniform_label_shift
