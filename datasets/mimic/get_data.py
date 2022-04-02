"""Implements dataloaders for generic MIMIC tasks."""
from tqdm import tqdm
from robustness.tabular_robust import add_tabular_noise
from robustness.timeseries_robust import add_timeseries_noise
import sys
import os
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))



def get_dataloader(task, batch_size=40, num_workers=1, train_shuffle=True, imputed_path='im.pk', flatten_time_series=False, tabular_robust=True, timeseries_robust=True):
    """Get dataloaders for MIMIC dataset.

    Args:
        task (int): Integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task.
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers to load data in. Defaults to 1.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        imputed_path (str, optional): Datafile location. Defaults to 'im.pk'.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        tabular_robust (bool, optional): Whether to apply tabular robustness as dataset augmentation or not. Defaults to True.
        timeseries_robust (bool, optional): Whether to apply timeseries robustness noises as dataset augmentation or not. Defaults to True.

    Returns:
        tuple: Tuple of training dataloader, validation dataloader, and test dataloader
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
    if task < 0:
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
    else:
        y = datafile['y_icd9'][:, task]
        le = len(y)
    datasets = [(X_s[i], X_t[i], y[i]) for i in range(le)]

    random.seed(10)

    random.shuffle(datasets)

    valids = DataLoader(datasets[0:le//10], shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(datasets[le//5:], shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size)

    tests = dict()
    tests['timeseries'] = []
    for noise_level in tqdm(range(11)):
        dataset_robust = copy.deepcopy(datasets[le//10:le//5])
        if tabular_robust:
            X_s_robust = add_tabular_noise([dataset_robust[i][0] for i in range(
                len(dataset_robust))], noise_level=noise_level/10)
        else:
            X_s_robust = [dataset_robust[i][0]
                          for i in range(len(dataset_robust))]
        if timeseries_robust:
            X_t_robust = add_timeseries_noise([[dataset_robust[i][1] for i in range(
                len(dataset_robust))]], noise_level=noise_level/10)[0]
        else:
            X_t_robust = [dataset_robust[i][1]
                          for i in range(len(dataset_robust))]
        y_robust = [dataset_robust[i][2] for i in range(len(dataset_robust))]
        if flatten_time_series:
            tests['timeseries'].append(DataLoader([(X_s_robust[i], X_t_robust[i].reshape(timestep*series_dim), y_robust[i])
                                       for i in range(len(y_robust))], shuffle=False, num_workers=num_workers, batch_size=batch_size))
        else:
            tests['timeseries'].append(DataLoader([(X_s_robust[i], X_t_robust[i], y_robust[i]) for i in range(
                len(y_robust))], shuffle=False, num_workers=num_workers, batch_size=batch_size))

    return trains, valids, tests
