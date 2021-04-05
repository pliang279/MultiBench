from typing import *
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Affectdataset(Dataset):
    
    def __init__(self, data:Dict, flatten_time_series:bool) -> None:
        self.dataset = data
        self.flatten = flatten_time_series

    def __getitem__(self, ind):
        if self.flatten:
            return [torch.tensor(self.dataset[i][ind]).flatten() \
                for i in self.dataset.keys() if i != 'id']
        else:
            return [torch.tensor(self.dataset[i][ind]) \
                for i in self.dataset.keys() if i != 'id']

    def __len__(self):
        return self.dataset['id'].shape[0]


def get_dataloader(
    filepath:str, batch_size:int=40, train_shuffle:bool=True,
    num_workers:int=8, flatten_time_series:bool=False)->DataLoader:

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    train = DataLoader(Affectdataset(alldata['train'], flatten_time_series), \
        shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    valid = DataLoader(Affectdataset(alldata['valid'], flatten_time_series), \
        shuffle=False, num_workers=num_workers, batch_size=batch_size)
    test = DataLoader(Affectdataset(alldata['test'], flatten_time_series), \
        shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train, valid, test
