import h5py
from typing import *

import torch
from torch.utils.data import Dataset, DataLoader


class IMDBDataset(Dataset):

    def __init__(self, file: h5py.File, start_ind: int, end_ind: int, vggfeature: bool = False) -> None:
        self.file = file
        self.start_ind = start_ind
        self.size = end_ind-start_ind
        self.vggfeature = vggfeature

    def __getitem__(self, ind):
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.file, 'r')
        text = self.dataset["features"][ind+self.start_ind]
        image = self.dataset["images"][ind+self.start_ind] if not self.vggfeature else \
            self.dataset["vgg_features"][ind+self.start_ind]
        label = self.dataset["genres"][ind+self.start_ind]

        return torch.tensor(text, dtype=torch.float32), torch.tensor(image, dtype=torch.float32), label

    def __len__(self):
        return self.size


def get_dataloader(
        path: str, num_workers: int = 8, train_shuffle: bool = True, batch_size: int = 40, vgg: bool = False) -> Tuple[DataLoader]:

    train_dataloader = DataLoader(IMDBDataset(path, 0, 15552, vgg),
                                  shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(IMDBDataset(path, 15552, 18160, vgg),
                                shuffle=False, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(IMDBDataset(path, 18160, 25959, vgg),
                                 shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader
