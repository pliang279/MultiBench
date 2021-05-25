import numpy as np


##############################################################################
# Audio
def audio_robustness(tests, noise_level=0.3, noises=None):
    if noises == None:
        noises = [additive_white_gaussian_noise, audio_random_dropout, audio_structured_dropout]
    robustness_tests = np.zeros(tests.shape)
    for i in range(len(tests)):
        if np.random.sample() <= noise_level:
            mode = np.random.randint(len(noises))
            robustness_tests[i] = noises[mode](tests[i], noise_level)
    return robustness_tests


def additive_white_gaussian_noise(signal, noise_level):
    # SNR = 10 * log((RMS of signal)^2 / (RMS of noise)^2)
    # RMS_s = np.sqrt(np.mean(signal*signal))
    # RMS_n = np.sqrt(RMS_s*RMS_s / (np.power(10, SNR/10)))
    noise = np.random.normal(0, noise_level, signal.shape[0])
    return signal + noise


def audio_structured_dropout(sig, p, step=10):
    # each consecutive time steps are chosen with probability p to be dropped
    res = [sig[i] for i in range(len(sig))]
    for i in range(len(res)-step+1):
        if (res[i] != 0) and np.random.random_sample() < p:
            for j in range(step):
                res[i+j] = 0
    return res

def audio_random_dropout(sig, p):
    return audio_structured_dropout(sig, 1, p)