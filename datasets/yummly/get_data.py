"""Implements dataloaders for the Yummly dataset."""

import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms

from nltk.tokenize import word_tokenize
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


def _load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _mk_label_map(label_set):
    '''
    Returns dictionary mapping raw label to integer (0-indexed)
    '''
    label_to_int = {}
    for i, label in enumerate(label_set):
        label_to_int[label] = i
    return label_to_int


def _tokenize_text(text, word2index):
    '''
    Args:
        text: string
    '''
    text_list = word_tokenize(text)

    new_text_list = []
    for word in text_list:
        if word in word2index:
            new_text_list.append(word2index[word])
        else:
            new_text_list.append(1)
    text_list = new_text_list
    return text_list


class YummlyDataset(Dataset):
    """Implements Yummly Data as a torch Dataset."""
    
    def __init__(self, data_dir, phase, width, height, word2index):
        """Initialize YummlyDataset.

        Args:
            data_dir (str): Data Directory
            phase (str): 'train' or 'valid', else all other data.
            width (int): Width to resize data to 
            height (int): Height to resize data to
            word2index (dictionary): Dictionary of word to indexes for tokenizer
        """
        fid_to_label = _load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
        fid_to_text = _load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
        fids = list(fid_to_label.keys())
        paths = [os.path.join(data_dir, 'images', 'img%s.jpg' % fid)
                 for fid in fids]
        label_to_int = _mk_label_map(set(fid_to_label.values()))
        targets = [label_to_int[fid_to_label[fid]] for fid in fids]
        num_labels = len(np.unique(targets))
        texts = [fid_to_text[fid] for fid in fids]
        num_train = int(len(paths)*0.8)
        num_valid = int(len(paths)*0.1)
        if phase == 'train':
            self.paths = paths[:num_train]
            self.targets = targets[:num_train]
            self.texts = texts[:num_train]
        elif phase == 'valid':
            self.paths = paths[num_train:num_train+num_valid]
            self.targets = targets[num_train:num_train+num_valid]
            self.texts = texts[num_train:num_train+num_valid]
        else:
            self.paths = paths[num_train+num_valid:]
            self.targets = targets[num_train+num_valid:]
            self.texts = texts[num_train+num_valid:]
        self.transform_image = transforms.Compose([lambda x: Image.open(x),
                                                   lambda x: x.resize(
                                                       (width, height)),
                                                   lambda x: np.reshape(
                                                       x, (width, height, 3)),
                                                   lambda x: np.transpose(
                                                       x, [2, 0, 1]),
                                                   lambda x: x/255.])
        self.transform_text = transforms.Compose(
            [lambda x: _tokenize_text(x, word2index)])

    def __getitem__(self, i):
        """Get item at index i.

        Args:
            i (int): Index of data

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): Tuple of (input modality 1, input modality 2, targets)
        """
        p = self.paths[i]
        t = self.texts[i]
        return self.transform_image(p), self.transform_input(t), self.targets[i]

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.paths)


def collate_yummly(batch):
    """Collate yummly data into batches.

    Args:
        batch (list(elements)): List of tuple(img, text, label)

    Returns:
        torch.Tensor: Collated tensors for imgs, texts, and labels in a batch.
    """
    new_imgs = []
    new_texts = []
    new_labels = []
    for (img, text, label) in batch:
        new_imgs.append(img)
        new_texts.append(torch.tensor(text))
        new_labels.append(label)
    new_imgs = torch.tensor(new_imgs)
    new_texts = pad_sequence(new_texts, batch_first=True, padding_value=0)
    new_labels = torch.tensor(new_labels)
    return new_imgs.float(), new_texts, new_labels


def get_dataloader(data_dir, batch_size=40, num_workers=8, train_shuffle=True):
    """Create train, test, and validation dataloaders.

    Args:
        data_dir (str): Data directory
        batch_size (int, optional): Batch size. Defaults to 40.
        num_workers (int, optional): Number of workers. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.

    Returns:
        tuple: Tuple of training dataloader, validation dataloader, test dataloader
    """
    word2index_path = os.path.join(data_dir, 'word2index.pkl')
    word2index = _load_pkl(word2index_path)
    valid_set = YummlyDataset(data_dir, 'valid', 128, 128, word2index)
    test_set = YummlyDataset(data_dir, 'test', 128, 128, word2index)
    train_set = YummlyDataset(data_dir, 'train', 128, 128, word2index)
    valids = DataLoader(valid_set, shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    tests = DataLoader(test_set, shuffle=False,
                       num_workers=num_workers, batch_size=batch_size)
    trains = DataLoader(train_set, shuffle=train_shuffle,
                        num_workers=num_workers, batch_size=batch_size)
    return trains, valids, tests
