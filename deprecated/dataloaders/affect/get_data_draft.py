import h5py
import pickle
import os
from typing import *

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def get_dataset(path: str, data_kind: str, embedding_file: str) -> Dataset:
    if data_kind == 'pkl' or 'sarcasm':
        f = load_pickle(path)
    elif data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        print('Wrong data kind!')

    word_embedding = h5py.File(embedding_file, 'r')

    return Affectdataset(f, data_kind, word_embedding)


def load_pickle(pickle_file: str) -> Dict:
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def get_dataloader(
        dataFolder: str, dataset: str, embedding: str = None,
        num_workers: int = 8, train_shuffle: bool = True, batch_size: int = 40) -> Tuple(DataLoader):

    cmu_data = ['mosi', 'mosi_unalign', 'mosei',
                'mosei_unalign', 'pom', 'pom_unalign']
    pkl_data = ['urfunny', 'deception']

    if dataset == 'sarcasm':
        datafile = dataset+'.pkl'
        file = os.path.join(dataFolder, datafile, embedding)
        dataset = get_dataset(file, 'sarcasm')
    elif dataset in cmu_data:
        datafile = dataset + '.hdf5'
        file = os.path.join(dataFolder, datafile, embedding)
        dataset = get_dataset(file, 'hdf5')
    elif dataset in pkl_data:
        datafile = dataset+'.pkl'
        file = os.path.join(dataFolder, datafile, embedding)
        dataset = get_dataset(file, 'pkl')
    else:
        print('Wrong Input!')

    size = len(dataset)
    train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                            [size//10, size//10, size-2*(size//10)], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(
        train_dataset, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class Affectdataset(Dataset):

    def __init__(self, file: Union(h5py.File, Dict), data_kind: str, embedding: h5py.File) -> None:
        self.dataset = file
        self.data_kind = data_kind
        self.modals = list(file.keys())
        self.index = list(self.modals[0].keys())
        self.word_embedding = embedding

    def __getitem__(self, ind):
        item_index = self.index[ind]
        features = []

        if self.data_kind == 'pkl':
            context_num = len(
                self.dataset['words'][item_index]['context_sentences'])
            for modal in self.modals:
                if modal == 'covarep' or 'open_face':
                    features.append(torch.tensor(
                        self.dataset[modal][item_index]['punchline_features']))
                    features.append([torch.tensor(self.dataset[modal][item_index]['context_features'][i])
                                     for i in range(context_num)])
                elif modal == 'words':
                    punchline_embedding = []
                    for embedding_ind in self.dataset[modal][item_index]['punchline_embedding_indexes']:
                        punchline_embedding.append(
                            self.word_embedding[embedding_ind])
                    features.append(torch.tensor(punchline_embedding))
                    context_embeddings = []
                    for context_ind in range(context_num):
                        context_embedding = []
                        for embedding_ind in self.dataset[modal][item_index]['context_embedding_indexes'][context_ind]:
                            context_embedding.append(
                                self.word_embedding[embedding_ind])
                        context_embeddings.append(
                            torch.tensor(context_embedding))
                    features.append(context_embeddings)
                elif modal == 'labels':
                    label = self.dataset[modal][item_index]
                else:
                    print('Features Not Included!')

        elif self.data_kind == 'sarcasm':
            features.append(torch.tensor(self.dataset['audio'][item_index]))
            # TO DO: can use pretrained word embedding
            label = self.dataset['text'][item_index]['sarcasm']

        elif self.data_kind == 'hdf5':
            for modal in self.modals:
                if 'label' not in modal.lower() and modal != 'words':
                    features.append(torch.tensor(
                        self.dataset[modal][item_index]['features']))
                elif modal == 'words':
                    features.append(
                        self.dataset[modal][item_index]['features'])
                    # TO DO: can use pretrained word embedding
                else:
                    label = self.dataset[modal][item_index]['features'][0][0]
                    # TO DO: decide the label we should use for pom

        return features, label

    def __len__(self):
        return len(self.index)
