"""Custom PyTorch samplers."""

import torch
from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices, shuffle):  # pylint: disable=W0231
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
