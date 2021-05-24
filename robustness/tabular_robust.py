import numpy as np


##############################################################################
# Tabular
def tabular_robustness(tests, noise_level=0.3, drop=True, swap=True):
    robust_tests = np.array(tests)
    if drop:
        robust_tests = drop_entry(robust_tests, noise_level)
    if swap:
        robust_tests = swap_entry(robust_tests, noise_level)
    return robust_tests


def drop_entry(data, p):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = 0
            else:
                data[i][j] = data[i][j]
    return data


def swap_entry(data, p):
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = data[i][j-1]
                data[i][j-1] = data[i][j]
    return data