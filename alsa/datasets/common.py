"""Utilities for accessing, batching and shifting datasets."""

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader
from torchvision import transforms
from alsa.config import home_dir
from alsa.datasets.utils import SubsetSampler
from alsa.datasets.shift_transforms import shift_dataset


class ActiveDataset():  # pylint: disable=R0902
    """Primary class for selective sampling logic."""
    def __init__(self,
                 dataset_cls,
                 label_cls,
                 domain_cls,
                 train_transforms,
                 test_transforms,
                 args,
                 moments=None):
        """Compute moments, build transformations, initialize torchvision."""
        self._args = args

        # Construct standard and training-time transforms
        transform = [
            transforms.ToTensor(),
        ]
        if moments:
            transform.append(transforms.Normalize(moments[0], moments[1]))

        transform_test = [*test_transforms, *transform]
        transform_test = transforms.Compose(transform_test)
        transform_train = [*train_transforms, *transform]
        transform_train = transforms.Compose(transform_train)

        # Build new torchvision datasets
        self._train_dataset = dataset_cls(root=home_dir + 'content/data',
                                          train=True,
                                          download=args.dataset != "nabirds",
                                          transform=transform_train)
        self.train_labels = label_cls(root=home_dir + 'content/data',
                                      train=True,
                                      download=args.dataset != "nabirds")
        self._test_dataset = dataset_cls(root=home_dir + 'content/data',
                                         train=False,
                                         download=args.dataset != "nabirds",
                                         transform=transform_test)
        self.test_labels = label_cls(root=home_dir + 'content/data',
                                     train=False,
                                     download=args.dataset != "nabirds")
        if args.dataset == "nabirds":
            self._train_dataset.change_class_map(args)
            self._test_dataset.change_class_map(args)
            self.train_labels.change_class_map(args)
            self.test_labels.change_class_map(args)

        if args.domainsep:
            traindomain = domain_cls(root=home_dir + 'content/data',
                                     train=True,
                                     download=args.dataset != "nabirds",
                                     domain=0)
            testdomain = domain_cls(root=home_dir + 'content/data',
                                    train=False,
                                    download=args.dataset != "nabirds",
                                    domain=1)
            self._domain_dataset = torch.utils.data.ConcatDataset(
                [traindomain, testdomain])

        # Compute facts
        self.test_labels_ls = []
        for l, _ in self.test_labels:
            self.test_labels_ls.append(l)

        self.label_space = np.unique(self.test_labels_ls)
        self.label_space.sort()
        self._num_cls = len(self.label_space)
        self._num_channels = self._train_dataset[0][0].shape[0]

        # Initialize list of split idxs
        self._warmstart_idxs = np.zeros((0, ), dtype=np.int32)
        self._initial_idxs = np.zeros((0, ), dtype=np.int32)
        self._online_idxs = np.zeros((0, ), dtype=np.int32)
        self._test_idxs = np.zeros((0, ), dtype=np.int32)

        # Initialize list of online idxs
        self._online_ptrs = np.zeros((0, ), dtype=np.int32)
        self.divide()

        self.label_weights = np.ones(args.num_cls, dtype=np.float32)

    def label_ptrs(self, ptrs):
        """Add online ptrs to the labeled set. Replaces old label set."""
        self._online_ptrs = np.array(ptrs, dtype=np.int32)

    def indices(self, split):
        """Return indices corresponding to specific split; positioned by ptr"""
        if split == "labeled":
            idxs = np.concatenate([
                self._warmstart_idxs, self._initial_idxs,
                self._online_idxs[self._online_ptrs]
            ])
        if split == "unlabeled":
            ignore_idxs = np.concatenate([
                self._warmstart_idxs, self._initial_idxs,
                self._online_idxs[self._online_ptrs]
            ])
            idxs = np.setdiff1d(np.arange(len(self._train_dataset)),
                                ignore_idxs,
                                assume_unique=True)
        if split == "test":
            idxs = self._test_idxs
        if split == "online":
            idxs = self._online_idxs
        if split == "initial":
            idxs = self._initial_idxs
        if split == "warmstart":
            idxs = self._warmstart_idxs
        if split == "all_train":
            idxs = np.arange(len(self._train_dataset))
        if split == "all_test":
            idxs = np.arange(len(self._test_dataset))
        return idxs

    def online_iterate(self,
                       ptrs,
                       batch_size,
                       shuffle,
                       label_only=False,
                       idxs=None,
                       split="online"):
        """Return data loader for training dataset"""
        # Select dataset
        if label_only:
            dataset = self.test_labels if split == "test" else self.train_labels
        else:
            dataset = self._test_dataset if split == "test" else self._train_dataset

        # Choose num workers
        num_workers = 0 if label_only else 16
        if self._args.dataset == "mnist":
            num_workers = 0

        # Grab indices
        if idxs is None:
            idxs = self.indices(split)[ptrs]

        # Build data loaders
        sampler = SubsetSampler(idxs, shuffle=shuffle)
        if batch_size <= 1:
            return DataLoader(dataset,
                              sampler=sampler,
                              num_workers=num_workers)
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=num_workers)

    def domain_iterate(self, batch_size, shuffle, label_only=False):
        """Return data loader for training dataset"""
        # Select dataset
        dataset = self._domain_dataset

        # Choose num workers
        num_workers = 0 if label_only else 16
        if self._args.dataset == "mnist":
            num_workers = 0

        # Grab indices
        idxs = np.concatenate([
            self.indices("online"),
            self.indices("test") + len(self._train_dataset)
        ])

        # Build data loaders
        sampler = SubsetSampler(idxs, shuffle=shuffle)
        if batch_size <= 1:
            return DataLoader(dataset,
                              sampler=sampler,
                              num_workers=num_workers)
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=num_workers)

    def iterate(self, batch_size, shuffle, label_only=False, split=None):
        """Return data loader for training dataset"""
        # Select dataset
        if label_only:
            dataset = self.test_labels if split == "test" else self.train_labels
        else:
            dataset = self._test_dataset if split == "test" else self._train_dataset

        # Choose num workers
        num_workers = 0 if label_only else 16

        # Grab indices
        idxs = self.indices(split)

        # Build data loaders
        sampler = SubsetSampler(idxs, shuffle=shuffle)
        if batch_size <= 1:
            return DataLoader(dataset,
                              sampler=sampler,
                              num_workers=num_workers)
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        return DataLoader(dataset,
                          batch_sampler=sampler,
                          num_workers=num_workers)

    def online_len(self):
        """Return length of online dataset split."""
        return len(self._online_idxs)

    def labeled_len(self):
        """Return length of all labeled dataset split."""
        return len(self._warmstart_idxs) + len(self._initial_idxs) + len(
            self._online_ptrs)

    def online_labeled_len(self):
        """Return length of online labeled dataset split."""
        return len(self._initial_idxs) + len(self._online_ptrs)

    def divide(self):
        """Divide train dataset into online, initial and warmstart splits."""
        # Split training set into exclusive warm start set and normal set
        warmstart_proportion = self._args.warmstart_ratio / \
            (1 + self._args.warmstart_ratio)
        split_loc = int(warmstart_proportion * len(self._train_dataset))
        print("Coarse source/target split:", split_loc,
              len(self._train_dataset) - split_loc)

        # Train idxs by label
        warmstart_idxs_by_label = {l: [] for l in self.label_space}
        online_idxs_by_label = {l: [] for l in self.label_space}
        num_added = 0
        for l, i in self.train_labels:
            if num_added < split_loc:
                warmstart_idxs_by_label[l].append(i)
            else:
                online_idxs_by_label[l].append(i)
            num_added += 1

        # Test idxs by label
        test_idxs_by_label = {
            label: list(np.where(self.test_labels_ls == label)[0])
            for label in self.label_space
        }
        for k in test_idxs_by_label:
            assert k in self.label_space

        self._warmstart_idxs, self._online_idxs, self._test_idxs = shift_dataset(
            self._args, warmstart_idxs_by_label, online_idxs_by_label,
            test_idxs_by_label)
        self._warmstart_idxs = np.array(self._warmstart_idxs, dtype=np.int32)
        self._online_idxs = np.array(self._online_idxs, dtype=np.int32)
        self._test_idxs = np.array(self._test_idxs, dtype=np.int32)

        # Extract non-exclusive initial training set
        split_loc = int(self._args.initial_prop * len(self._online_idxs))
        self._initial_idxs = self._online_idxs[:split_loc]
        self._online_idxs = self._online_idxs[split_loc:]

        # Transform online and testing datasets
        print("Warm start/initial/online/test split:",
              len(self._warmstart_idxs), len(self._initial_idxs),
              len(self._online_idxs), len(self._test_idxs))

    def set_weight(self, weight_map):
        """Add importance weighting to datasets"""
        if self._args.reweight:
            self._train_dataset.add_weight(weight_map, self.indices("labeled"))
        else:
            self._train_dataset.add_weight(weight_map,
                                           self.indices("warmstart"))
        for i in range(self._args.num_cls):
            self.label_weights[i] = weight_map[i]
