import h5py
from typing import *

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def get_dataloader(
    path:str, num_workers:int=8, train_shuffle:bool=True, batch_size:int=40)->Tuple(DataLoader):
    
    f = h5py.File(path, 'r')
    dataset = IMDBdataset(f)

    size = len(dataset)
    train_dataset, val_dataset, test_dataset = random_split(dataset, \
        [size//10, size//10, size-2*(size//10)], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(train_dataset, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class IMDBdataset(Dataset):
    
    def __init__(self, file:h5py.File, ) -> None:
        self.dataset = file
        self.modals = list(file.keys())
        self.index = list(self.modals[0].keys())

    def __getitem__(self, ind):
        item_index = self.index[ind]
        features = []

        for modal in self.modals:
            if 'label' not in modal.lower() and modal != 'words':
                features.append(torch.tensor(self.dataset[modal][item_index]['features']))
            else:
                label = self.dataset[modal][item_index]['features'][0][0]

        return features, label

    def __len__(self):
        return len(self.index)

