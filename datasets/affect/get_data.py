from typing import *
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class Affectdataset(Dataset):
    
    def __init__(self, data:Dict, flatten_time_series:bool, aligned:bool=True, task:str=None) -> None:
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task

    def __getitem__(self, ind):
        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])
        if self.aligned:
            start = text.nonzero()[0][0]
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()
        label = torch.tensor(self.dataset['labels'][ind]).float().round() if self.task == "classification" else\
            torch.tensor(self.dataset['labels'][ind]).float()
        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind,\
                    torch.tensor(self.dataset['labels'][ind]).float()]
        else:
            return [vision, audio, text, ind, torch.tensor(self.dataset['labels'][ind]).float()]

    def __len__(self):
        return self.dataset['id'].shape[0]


def get_dataloader(
    filepath:str, batch_size:int=40, train_shuffle:bool=True,
    num_workers:int=8, flatten_time_series:bool=False)->DataLoader:

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    train = DataLoader(Affectdataset(alldata['train'], flatten_time_series), \
        shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, \
        collate_fn=process)
    valid = DataLoader(Affectdataset(alldata['valid'], flatten_time_series), \
        shuffle=False, num_workers=num_workers, batch_size=batch_size, \
        collate_fn=process)
    test = DataLoader(Affectdataset(alldata['test'], flatten_time_series), \
        shuffle=False, num_workers=num_workers, batch_size=batch_size, \
        collate_fn=process)

    return train, valid, test


def process(inputs:List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []
    
    for i in range(len(inputs[0])-2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        processed_input.append(pad_sequence(feature, batch_first=True))
    
    for sample in inputs:
        inds.append(sample[-2])
        labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
        torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)
