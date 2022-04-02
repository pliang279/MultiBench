"""Implements data loaders for the multitask formulation of MIMIC."""
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
# task -1 and task 7 only


def get_dataloader(batch_size=40, num_workers=1, train_shuffle=True, imputed_path='im.pk', flatten_time_series=False):
    """Generate dataloader for multi-task setup, using only tasks -1 and 7.

    Args:
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers to load data in. Defaults to 1.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        imputed_path (str, optional): Datafile location. Defaults to 'im.pk'.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.

    Returns:
        tupe: Tuple of training dataloader, validation dataloader, and test dataloader.
    """
    f = open(imputed_path, 'rb')
    datafile = pickle.load(f)
    f.close()
    X_t = datafile['ep_tdata']
    X_s = datafile['adm_features_all']

    X_t[np.isinf(X_t)] = 0
    X_t[np.isnan(X_t)] = 0
    X_s[np.isinf(X_s)] = 0
    X_s[np.isnan(X_s)] = 0

    X_s_avg = np.average(X_s, axis=0)
    X_s_std = np.std(X_s, axis=0)
    X_t_avg = np.average(X_t, axis=(0, 1))
    X_t_std = np.std(X_t, axis=(0, 1))

    for i in range(len(X_s)):
        X_s[i] = (X_s[i]-X_s_avg)/X_s_std
        for j in range(len(X_t[0])):
            X_t[i][j] = (X_t[i][j]-X_t_avg)/X_t_std

    static_dim = len(X_s[0])
    timestep = len(X_t[0])
    series_dim = len(X_t[0][0])
    if flatten_time_series:
        X_t = X_t.reshape(len(X_t), timestep*series_dim)
    if True:
        y = datafile['adm_labels_all'][:, 1]
        admlbl = datafile['adm_labels_all']
        le = len(y)
        for i in range(0, le):
            if admlbl[i][1] > 0:
                y[i] = 1
            elif admlbl[i][2] > 0:
                y[i] = 2
            elif admlbl[i][3] > 0:
                y[i] = 3
            elif admlbl[i][4] > 0:
                y[i] = 4
            elif admlbl[i][5] > 0:
                y[i] = 5
            else:
                y[i] = 0
    y2 = datafile['y_icd9'][:, 7]
    datasets = [(X_s[i], X_t[i], y[i], y2[i]) for i in range(le)]

    random.seed(10)

    random.shuffle(datasets)

    valids = DataLoader(datasets[0:le//10], shuffle=False,
                        num_workers=num_workers, batch_size=40)
    tests = DataLoader(datasets[le//10:le//5], shuffle=False,
                       num_workers=num_workers, batch_size=40)
    trains = DataLoader(datasets[le//5:], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size)
    return trains, valids, tests
