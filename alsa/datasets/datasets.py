"""Define datasets API for accessing interactive dataset objects."""

import numpy as np
from torchvision import datasets, transforms
from alsa.datasets.common import ActiveDataset
from alsa.datasets.nabirds import NABirds
from alsa.config import cifar100_mean, cifar100_std, nabirds_mean, nabirds_std


def dataset_factory(dataset_cls):
    """Create wrapper of torchvision dataset to also return indices."""
    class DomainDataset(dataset_cls):  # pylint: disable=R0903
        """Wrapping of torchvision dataset"""
        def __init__(self, *args, domain=None, **kargs):
            super().__init__(*args, **kargs)
            self.domain = domain

        def __getitem__(self, index):
            """Modifies get item by also returning idx."""
            img, _ = super().__getitem__(index)
            return img, self.domain

    class WeightedDataset(dataset_cls):
        """Wrapping of torchvision dataset"""
        def __init__(self, *args, **kargs):
            super().__init__(*args, **kargs)
            self.weights = np.ones(self.__len__(), dtype=np.float32)

        def __getitem__(self, index):
            """Modifies get item by also returning idx."""
            img, target = super().__getitem__(index)
            weight = self.weights[index]
            return img, target, weight

        def add_weight(self, weight_map, indices):
            """Add importance weights to dataset. By default ones."""
            if self.targets is None:
                for i in indices:
                    self.weights[i] = weight_map[self.get_target(i)]
            else:
                for i in indices:
                    self.weights[i] = weight_map[self.targets[i]]

    class LabelsDataset(dataset_cls):  # pylint: disable=R0903
        """Wrapping of torchvision dataset"""
        def __getitem__(self, index):
            """Modifies get item by also returning idx."""
            try:
                if self.targets is None:
                    target = self.get_target(index)
                else:
                    target = int(self.targets[index])
            except AttributeError:
                target = self.label_map[self.data.iloc[index].target]

            return target, index

    return WeightedDataset, LabelsDataset, DomainDataset


def get_datasets(args):
    """Grab MNIST, CIFAR10, and CIFAR100 datasets"""
    if args.dataset == "mnist":
        dataset_cls = datasets.MNIST
        train_transforms = []
        test_transforms = []
        moments = ((0.1307, ), (0.3081, ))
        args.num_cls = 10
    elif args.dataset == "cifar":
        dataset_cls = datasets.CIFAR10
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
        test_transforms = []
        moments = None
        args.num_cls = 10
    elif args.dataset == "cifar100":
        dataset_cls = datasets.CIFAR100
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
        test_transforms = []
        moments = (cifar100_mean, cifar100_std)
        args.num_cls = 100
    elif args.dataset == "nabirds":
        dataset_cls = NABirds
        test_transforms = [
            transforms.Resize(200),
            transforms.CenterCrop(150),
        ]
        train_transforms = [
            transforms.Resize(200),
            transforms.RandomCrop(150, pad_if_needed=True),
        ]
        if args.nabirdstype == "child":
            args.num_cls = 22
        elif args.nabirdstype == "grand":
            args.num_cls = 228
        else:
            raise ValueError()
        moments = (nabirds_mean, nabirds_std)
    else:
        raise ValueError()

    dataset_classes = dataset_factory(dataset_cls)
    return ActiveDataset(dataset_classes[0], dataset_classes[1],
                         dataset_classes[2], train_transforms, test_transforms,
                         args, moments)
