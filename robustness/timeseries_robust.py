"""Implements timeseries transformations."""
import numpy as np


##############################################################################
# Time-Series
def add_timeseries_noise(tests, noise_level=0.3, gaussian_noise=True, rand_drop=True, struct_drop=True):
    """
    Add various types of noise to timeseries data.
    
    :param noise_level: Standard deviation of gaussian noise, and drop probability in random drop and structural drop
    :param gauss_noise:  Add Gaussian noise to the time series ( default: True )
    :param rand_drop: Add randomized dropout to the time series ( default: True )
    :param struct_drop: Add randomized structural dropout to the time series ( default: True )
    """
    # robust_tests = np.array(tests)
    robust_tests = tests
    if gaussian_noise:
        robust_tests = white_noise(robust_tests, noise_level)
    if rand_drop:
        robust_tests = random_drop(robust_tests, noise_level)
    if struct_drop:
        robust_tests = structured_drop(robust_tests, noise_level)
    return robust_tests


def white_noise(data, p):
    """Add noise sampled from zero-mean Gaussian with standard deviation p at every time step.
    
    :param data: Data to process.
    :param p: Standard deviation of added Gaussian noise.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            data[i][time] += np.random.normal(0, p)
    return data




def random_drop(data, p):
    """Drop each time series entry independently with probability p.
    
    :param data: Data to process.
    :param p: Probability to drop feature.
    """
    for i in range(len(data)):
        data[i] = _random_drop_helper(data[i], p, len(np.array(data).shape))
    return data


def _random_drop_helper(data, p, level):
    """
    Helper function that implements random drop for 2-/higher-dimentional timeseris data.

    :param data: Data to process.
    :param p: Probability to drop feature.
    :param level: Dimensionality.
    """
    if level == 2:
        for i in range(len(data)):
            if np.random.random_sample() < p:
                data[i] = 0
        return data
    else:
        for i in range(len(data)):
            data[i] = _random_drop_helper(data[i], p, level - 1)
        return data


def structured_drop(data, p):
    """Drop each time series entry independently with probability p, but drop all modalities if you drop an element.
    
    :param data: Data to process.
    :param p: Probability to drop entire element of time series.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][time] = np.zeros(data[i][time].shape)
    return data
