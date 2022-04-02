"""Implements tabular data transformations."""
import numpy as np


##############################################################################
# Tabular
def add_tabular_noise(tests, noise_level=0.3, drop=True, swap=True):
    """
    Add various types of noise to tabular data.

    :param noise_level: Probability of randomly applying noise to each element.
    :param drop: Drop elements with probability `noise_level`
    :param swap: Swap elements with probability `noise_level`
    """
    
    robust_tests = np.array(tests)
    if drop:
        robust_tests = drop_entry(robust_tests, noise_level)
    if swap:
        robust_tests = swap_entry(robust_tests, noise_level)
    return robust_tests


def drop_entry(data, p):
    """
    Randomly drop elements in `data` with probability `p`
    
    :param data: Data to drop elements from.
    :param p: Probability of dropping elements.
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = 0
            else:
                data[i][j] = data[i][j]
    return data


def swap_entry(data, p):
    """
    Randomly swap adjacent elements in `data` with probability `p`.
    
    :param data: Data to swap elems.
    :param p: Probability of swapping elements.
    """
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = data[i][j-1]
                data[i][j-1] = data[i][j]
    return data
