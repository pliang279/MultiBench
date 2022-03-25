"""Implements processforce, which truncates force readings to a window size."""

import torch
import numpy as np

class ProcessForce(object):
    """Truncate a time series of force readings with a window size.
    Args:
        window_size (int): Length of the history window that is
            used to truncate the force readings
    """

    def __init__(self, window_size, key='force', tanh=False):
        """Initialize ProcessForce object.

        Args:
            window_size (int): Windows size
            key (str, optional): Key where data is stored. Defaults to 'force'.
            tanh (bool, optional): Whether to apply tanh to output or not. Defaults to False.
        """
        assert isinstance(window_size, int)
        self.window_size = window_size
        self.key = key
        self.tanh = tanh

    def __call__(self, sample):
        """Get data from sample."""
        force = sample[self.key]
        force = force[-self.window_size:]
        if self.tanh:
            force = np.tanh(force)  # remove very large force readings
        sample[self.key] = force.transpose()
        return sample
