"""Implements audio transformations."""
import numpy as np


##############################################################################
# Audio
def add_audio_noise(tests, noise_level=0.3, noises=None):
    """
    Add various types of noise to audio data.

    :param noise_level: Probability of randomly applying noise to each audio signal, and standard deviation for gaussian noise, and structured dropout probability.
    :param noises: list of noises to add. # TODO: Change this to use either a list of enums or if statements.
    """
    if noises is None:
        noises = [additive_white_gaussian_noise,
                  audio_random_dropout, audio_structured_dropout]
    robustness_tests = np.zeros(tests.shape)
    for i in range(len(tests)):
        if np.random.sample() <= noise_level:
            mode = np.random.randint(len(noises))
            robustness_tests[i] = noises[mode](tests[i], noise_level)
    return robustness_tests


def additive_white_gaussian_noise(signal, noise_level):
    """
    Add gaussian white noise to audio signal.

    :param signal: Audio signal to permute.
    :param noise_level: standard deviation of the gaussian noise.
    """
    # SNR = 10 * log((RMS of signal)^2 / (RMS of noise)^2)
    # RMS_s = np.sqrt(np.mean(signal*signal))
    # RMS_n = np.sqrt(RMS_s*RMS_s / (np.power(10, SNR/10)))
    noise = np.random.normal(0, noise_level, signal.shape[0])
    return signal + noise


def audio_structured_dropout(signal, p, step=10):
    """
    Randomly drop signal for `step` time steps.

    :param signal: Audio signal to permute.
    :param p: Dropout probability.
    :param step: Number of time steps to drop the signal.
    """
    res = [signal[i] for i in range(len(signal))]
    for i in range(len(res)-step+1):
        if (res[i] != 0) and np.random.random_sample() < p:
            for j in range(step):
                res[i+j] = 0
    return res


def audio_random_dropout(sig, p):
    """
    Randomly drop the signal for a single time step.

    :param signal: Audio signal to transform.
    :param p: Dropout probability.
    """
    return audio_structured_dropout(sig, 1, p)
