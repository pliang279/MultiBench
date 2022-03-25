"""Implements another utility for this dataset."""
import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        """Initialize ToTensor object."""
        self.device = device

    def __call__(self, sample):
        """Convert sample argument from ndarray with H,W,C dimensions to a tensor with C,H,W dimensions."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        # transpose flow into 2 x H x W
        for k in sample.keys():
            if k.startswith('flow'):
                sample[k] = sample[k].transpose((2, 0, 1))

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            if self.device is None:
                # torch.tensor(v, device = self.device, dtype = torch.float32)
                new_dict[k] = torch.FloatTensor(v)
            else:
                new_dict[k] = torch.from_numpy(v).float()

        return new_dict
