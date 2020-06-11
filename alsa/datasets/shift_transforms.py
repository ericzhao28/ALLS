"""Resources for shifting datasets."""

import random
import math
import numpy as np


def uniform_dataset(idxs_by_label, size, replace=True, verbose=True):
    """Uniformly distribute dataset."""

    # Group data points by label
    if replace:
        # Sample size points from each
        cls_size = math.ceil(size / len(idxs_by_label))
        for l in idxs_by_label:
            if idxs_by_label[l]:
                idxs = np.random.randint(
                    0, len(idxs_by_label[l]), size=(cls_size,))
                idxs_by_label[l] = [idxs_by_label[l][i] for i in idxs]
    else:
        # How many data points should we keep in classes?
        cls_size = min([len(v) for v in idxs_by_label.values()])
        # Handle excess points
        if cls_size * len(idxs_by_label) > size:
            cls_size = math.ceil(size / len(idxs_by_label))
        # Cut classes data points
        for l in idxs_by_label:
            idxs_by_label[l] = idxs_by_label[l][:cls_size]

    cls_composition = [len(idxs_by_label[k])
                       for k in sorted(idxs_by_label.keys())]
    cls_composition = np.array(cls_composition, dtype=np.float32)
    if verbose:
        print("Class composition: ", cls_composition)

    # Build new dataset
    dataset = []
    for v in idxs_by_label.values():
        dataset += v
    random.shuffle(dataset)
    if verbose:
        print("Full dataset size: ", len(dataset))

    return dataset


def tweak_dataset(idxs_by_label, size, args, replace=True, verbose=True):
    """Adjust the likelihood of one specific class"""

    if replace:
        # Compute tweak size
        tweak_size = math.ceil(args.tweak_prop * size)
        other_size = math.ceil(size * (1 - args.tweak_prop) /
                               (len(idxs_by_label) - 1))
        # Cut classes data points
        for l in idxs_by_label:
            if idxs_by_label[l]:
                if l == args.tweak_label:
                    idxs = np.random.randint(
                        0, len(idxs_by_label[l]), size=(tweak_size,))
                else:
                    idxs = np.random.randint(
                        0, len(idxs_by_label[l]), size=(other_size,))
                idxs_by_label[l] = [idxs_by_label[l][i] for i in idxs]
    else:
        # How many data points should we keep in classes?
        new_size = len(idxs_by_label[args.tweak_label]) / args.tweak_prop
        if new_size > size:
            new_size = size
        other_size = math.ceil(new_size * (1 - args.tweak_prop) /
                               (len(idxs_by_label) - 1))
        # Cut classes data points
        for l in idxs_by_label:
            if l != args.tweak_label:
                idxs_by_label[l] = idxs_by_label[l][:other_size + 1]

    cls_composition = [len(idxs_by_label[k])
                       for k in sorted(idxs_by_label.keys())]
    cls_composition = np.array(cls_composition, dtype=np.float32)
    if verbose:
        print("Class composition: ", cls_composition)

    # Build new dataset
    dataset = []
    for v in idxs_by_label.values():
        dataset += v
    random.shuffle(dataset)
    if verbose:
        print("Full dataset size: ", len(dataset))

    return dataset


def dirichlet_dataset(
        idxs_by_label,
        size,
        args,
        replace=True,
        verbose=True,
        distribution=None):
    """Shift dataset with prior sampled from Dirichlet distribution."""

    # Compute distribution if not provided
    if distribution is None:
        distribution = np.random.dirichlet(
            [args.dirichlet_alpha] * len(idxs_by_label), size=())

    # Group data points by label
    if replace:
        for l in idxs_by_label:
            cls_size = math.ceil(size * distribution[l])
            if not cls_size:
                cls_size = 1
            if idxs_by_label[l]:
                idxs = np.random.randint(
                    0, len(idxs_by_label[l]), size=(cls_size,))
                idxs_by_label[l] = [idxs_by_label[l][i] for i in idxs]
    else:
        # Find resulting dataset size
        dataset_sizes = []
        for l in idxs_by_label.items():
            dataset_sizes.append(len(idxs_by_label[l]) / distribution[l])
        new_size = min(dataset_sizes)
        # Handle excess points
        if new_size > size:
            new_size = size
        # Cut classes data points
        for l in idxs_by_label:
            cls_size = math.ceil(new_size * distribution[l])
            if not cls_size:
                cls_size = 1
            idxs_by_label[l] = idxs_by_label[l][:cls_size]

    cls_composition = [len(idxs_by_label[k])
                       for k in sorted(idxs_by_label.keys())]
    cls_composition = np.array(cls_composition, dtype=np.float32)
    if verbose:
        print("Class composition: ", cls_composition)

    # Build new dataset
    dataset = []
    for v in idxs_by_label.values():
        dataset += v
    random.shuffle(dataset)
    if verbose:
        print("Full dataset size: ", len(dataset))

    return dataset, distribution


def shift_dataset(args, warmstart_idxs_by_label, online_idxs_by_label, test_idxs_by_label):
    """Shift source and target according to arg rules."""

    # Transform datasets
    warmstart_size = int(args.dataset_cap
                         * (args.warmstart_ratio / (args.warmstart_ratio + 1)))
    online_size = args.dataset_cap - warmstart_size
    test_size = online_size
    if args.dataset == "nabirds":
        test_size = 5000
    # test_size = sum([len(x) for x in test_idxs_by_label.values()])

    if args.shift_strategy == "dirichlet":
        # Dirichlet source, uniform target/test
        warmstart_idxs, p = dirichlet_dataset(
            warmstart_idxs_by_label, warmstart_size, args)
        online_idxs = uniform_dataset(online_idxs_by_label, online_size)
        test_idxs = uniform_dataset(test_idxs_by_label, test_size)
    elif args.shift_strategy == "tweak":
        # Tweak source, uniform target/test
        warmstart_idxs = tweak_dataset(
            warmstart_idxs_by_label, warmstart_size, args)
        online_idxs = uniform_dataset(online_idxs_by_label, online_size)
        test_idxs = uniform_dataset(test_idxs_by_label, test_size)
    elif args.shift_strategy == "trulynone":
        warmstart_idxs = np.concatenate(list(warmstart_idxs_by_label.values()))
        np.random.shuffle(warmstart_idxs)
        warmstart_idxs = warmstart_idxs[:warmstart_size]
        online_idxs = np.concatenate(list(online_idxs_by_label.values()))
        np.random.shuffle(online_idxs)
        online_idxs = online_idxs[:online_size]

        print("warmstart", [len(x) for x in warmstart_idxs_by_label.values()])
        print("online", [len(x) for x in online_idxs_by_label.values()])
        test_idxs = uniform_dataset(test_idxs_by_label, test_size)

    elif args.shift_strategy == "trulyregion":
        warmstart_idxs = np.concatenate(list(warmstart_idxs_by_label.values()))
        np.random.shuffle(warmstart_idxs)
        warmstart_idxs = warmstart_idxs[:warmstart_size]

        print("warmstart", [len(x) for x in warmstart_idxs_by_label.values()])
        online_idxs, p = dirichlet_dataset(
            online_idxs_by_label, online_size, args)
        test_idxs, p = dirichlet_dataset(
            test_idxs_by_label, test_size, args, distribution=p)

    elif args.shift_strategy == "none":
        # Uniform source, uniform target/test
        warmstart_idxs = uniform_dataset(
            warmstart_idxs_by_label, warmstart_size)
        online_idxs = uniform_dataset(online_idxs_by_label, online_size)
        test_idxs = uniform_dataset(test_idxs_by_label, test_size)
    elif args.shift_strategy == "dirichlet_target":
        # Uniform source, dirichlet target/test
        warmstart_idxs = uniform_dataset(
            warmstart_idxs_by_label, warmstart_size)
        online_idxs, p = dirichlet_dataset(
            online_idxs_by_label, online_size, args)
        test_idxs, p = dirichlet_dataset(
            test_idxs_by_label, test_size, args, distribution=p)
    elif args.shift_strategy == "dirichlet_online_target":
        # Uniform source, dirichlet target/test
        warmstart_idxs = uniform_dataset(
            warmstart_idxs_by_label, warmstart_size)
        online_idxs = uniform_dataset(
            online_idxs_by_label, online_size)
        test_idxs, p = dirichlet_dataset(
            test_idxs_by_label, test_size, args)
    elif args.shift_strategy == "dirichlet_identical":
        # Dirichlet source, dirichlet target/test (identical p)
        warmstart_idxs, p = dirichlet_dataset(
            warmstart_idxs_by_label, warmstart_size, args)
        online_idxs, p = dirichlet_dataset(
            online_idxs_by_label, online_size, args, distribution=p)
        test_idxs, p = dirichlet_dataset(
            test_idxs_by_label, test_size, args, distribution=p)
    elif args.shift_strategy == "dirichlet_mix":
        # Dirichlet source, dirichlet target/test (different p)
        warmstart_idxs, _ = dirichlet_dataset(
            warmstart_idxs_by_label, warmstart_size, args)
        online_idxs, p = dirichlet_dataset(
            online_idxs_by_label, online_size, args)
        test_idxs, p = dirichlet_dataset(
            test_idxs_by_label, test_size, args, distribution=p)
    elif args.shift_strategy == "dirichlet_online":
        # Dirichlet source, dirichlet target/test (different p)
        warmstart_idxs, p = dirichlet_dataset(
            warmstart_idxs_by_label, warmstart_size, args)
        online_idxs, _ = dirichlet_dataset(
            online_idxs_by_label, online_size, args)
        test_idxs, _ = dirichlet_dataset(
            test_idxs_by_label, test_size, args, distribution=p)
    elif args.shift_strategy == "dirichlet_online_source":
        # Dirichlet source, dirichlet target/test (different p)
        warmstart_idxs = uniform_dataset(
            warmstart_idxs_by_label, warmstart_size, args)
        online_idxs, _ = dirichlet_dataset(
            online_idxs_by_label, online_size, args)
        test_idxs = uniform_dataset(
            test_idxs_by_label, test_size, args)
    else:
        raise ValueError()
    return warmstart_idxs, online_idxs, test_idxs
