import numpy as np


##############################################################################
# Time-Series
def timeseries_robustness(tests, noise_level=0.3, noise=True, rand_drop=True, struct_drop=True):
    # robust_tests = np.array(tests)
    robust_tests = tests
    if noise:
        robust_tests = white_noise(robust_tests, noise_level)
    if rand_drop:
        robust_tests = random_drop(robust_tests, noise_level)
    if struct_drop:
        robust_tests = structured_drop(robust_tests, noise_level)  
    return robust_tests


# add noise sampled from zero-mean Gaussian with standard deviation p at every time step
def white_noise(data, p):
    for i in range(len(data)):
        for time in range(len(data[i])):
            data[i][time] += np.random.normal(0, p)
    return data

# each entry is dropped independently with probability p
def random_drop(data, p):
    for i in range(len(data)):
        data[i] = random_drop_helper(data[i], p, len(np.array(data).shape))
    return data

def random_drop_helper(data, p, level):
    if level == 2:
        for i in range(len(data)):
            if np.random.random_sample() < p:
                data[i] = 0
        return data
    else:
        for i in range(len(data)):
            data[i] = random_drop_helper(data[i], p, level - 1)
        return data


# independently for each modality, each time step is chosen with probability p 
# at which all feature dimensions are dropped
def structured_drop(data, p):
    for i in range(len(data)):
        for time in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][time] = np.zeros(data[i][time].shape)
    return data