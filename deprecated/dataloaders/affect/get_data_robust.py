from robustness.timeseries_robust import add_timeseries_noise
from robustness.text_robust import add_text_noise
from types import new_class
from typing import *
import h5py
import pickle
import os
import sys
import re
from collections import defaultdict

import numpy as np
import torch
import torchtext as text
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))


class Affectdataset(Dataset):

    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None) -> None:
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task

    def __getitem__(self, ind):
        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])
        if self.aligned:
            try:
                start = text.nonzero()[0][0]
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()
        label = torch.tensor(self.dataset['labels'][ind]).round().long()+3 if self.task == "classification" else\
            torch.tensor(self.dataset['labels'][ind]).float()
        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind,
                    label]
        else:
            return [vision, audio, text, ind, label]

    def __len__(self):
        return self.dataset['vision'].shape[0]


def get_rawtext(path, data_kind, vids):
    if data_kind == 'pkl' or data_kind == 'sarcasm':
        f = load_pickle(path)
    elif data_kind == 'hdf5':
        f = h5py.File(path, 'r')
        text_data = []
        new_vids = []
        count = 0
        for vid in vids:
            text = []
            (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
            vid_id = '{}[{}]'.format(id, seg)
            # TODO: fix 31 missing entries
            try:
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            except:
                print("missing", vid, vid_id)
        return text_data, new_vids
    else:
        print('Wrong data kind!')


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


def get_word2id(text_data, vids):
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['unk']
    data_processed = dict()
    for i, segment in enumerate(text_data):
        words = []
        _words = segment.split()
        for word in _words:
            words.append(word2id[word])
        words = np.asarray(words)
        data_processed[vids[i]] = words

    def return_unk():
        return UNK
    word2id.default_factory = return_unk
    return data_processed, word2id


def get_word_embeddings(word2id, save=False):
    vec = text.vocab.GloVe(name='840B', dim=300)
    tokens = []
    for w, _ in word2id.items():
        tokens.append(w)
    
    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
    return ret


def glove_embeddings(text_data, vids, paddings=50):
    data_prod, w2id = get_word2id(text_data, vids)
    word_embeddings_looks_up = get_word_embeddings(w2id)
    looks_up = word_embeddings_looks_up.numpy()
    embedd_data = []
    for vid in vids:
        d = data_prod[vid]
        tmp = []
        look_up = [looks_up[x] for x in d]
        # Padding with zeros at the front
        # TODO: fix some segs have more than 50 words
        for i in range(paddings-len(d)):
            tmp.append(np.zeros(300,))
        for x in d:
            tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:
        
        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


def drop_entry(dataset):
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def get_dataloader(
        filepath: str, dataFolder: str, dataset: str, batch_size: int = 40, train_shuffle: bool = True, num_workers: int = 8, flatten_time_series: bool = False, task=None) -> DataLoader:

    cmu_data = ['mosi', 'mosi_unalign', 'mosei',
                'mosei_unalign', 'pom', 'pom_unalign']
    pkl_data = ['urfunny', 'deception']
    vids = []

    with open(filepath, "rb") as f:
        alldata = pickle.load(f)
    if dataset == 'sarcasm':
        datafile = dataset+'.pkl'
        file = os.path.join(dataFolder, datafile)
        rawtext = get_rawtext(file, 'sarcasm')
    elif dataset in cmu_data:
        datafile = dataset + '.hdf5'
        vids = [id[0].decode('UTF-8') for id in alldata['test']['id']]
        file = os.path.join(dataFolder, datafile)
        rawtext, vids = get_rawtext(file, 'hdf5', vids)
    elif dataset in pkl_data:
        datafile = dataset+'.pkl'
        file = os.path.join(dataFolder, datafile)
        rawtext = get_rawtext(file, 'pkl')
    else:
        print('Wrong Input!')

    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    train = DataLoader(Affectdataset(alldata['train'], flatten_time_series, task=task),
                       shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, collate_fn=process)
    valid = DataLoader(Affectdataset(alldata['valid'], flatten_time_series, task=task),
                       shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process)

    # Add text noises
    robust_text = []
    for i in range(10):
        test = dict()
        test['vision'] = alldata['test']["vision"]
        test['audio'] = alldata['test']["audio"]
        test['text'] = glove_embeddings(
            add_text_noise(rawtext, noise_level=i/10), vids)
        test['labels'] = alldata['test']["label"]
        test = drop_entry(test)
        robust_text.append(DataLoader(Affectdataset(test, flatten_time_series, task=task),
                           shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process))

    # Add visual noises
    robust_vision = []
    for i in range(10):
        test = dict()
        test['vision'] = add_timeseries_noise(
            [alldata['test']['vision']], noise_level=i/10, rand_drop=False, struct_drop=False)
        test['audio'] = alldata['test']["audio"]
        test['text'] = alldata['test']['text']
        test['labels'] = alldata['test']["label"]
        test = drop_entry(test)
        robust_vision.append(DataLoader(Affectdataset(test, flatten_time_series, task=task),
                             shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process))

    # Add audio noises
    robust_audio = []
    for i in range(10):
        test = dict()
        test['vision'] = alldata['test']["vision"]
        test['audio'] = add_timeseries_noise(
            [alldata['test']['audio']], noise_level=i/10, rand_drop=False)
        test['text'] = alldata['test']['text']
        test['labels'] = alldata['test']["label"]
        test = drop_entry(test)
        robust_audio.append(DataLoader(Affectdataset(test, flatten_time_series, task=task),
                            shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process))

    # Add timeseries noises
    for i, text in enumerate(robust_text):
        alldata_test = add_timeseries_noise(
            [alldata['test']['vision'], alldata['test']['audio'], text], noise_level=i/10)
        test.append(alldata_test)

    robust_timeseries = []
    alldata['test'] = drop_entry(alldata['test'])
    for i in range(10):
        robust_timeseries_tmp = add_timeseries_noise(
            [alldata['test']['vision'], alldata['test']['audio'], alldata['test']['text']], noise_level=i/10)
        test = dict()
        test['vision'] = robust_timeseries_tmp[0]
        test['audio'] = robust_timeseries_tmp[1]
        test['text'] = robust_timeseries_tmp[2]
        test['labels'] = alldata['test']['labels']
        robust_timeseries.append(DataLoader(Affectdataset(test, flatten_time_series, task=task),
                                 shuffle=False, num_workers=num_workers, batch_size=batch_size, collate_fn=process))
    return train, valid, robust_text, robust_vision, robust_audio, robust_timeseries


def process(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0])-2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(
            torch.as_tensor([v.size(0) for v in feature]))
        processed_input.append(pad_sequence(feature, batch_first=True))

    for sample in inputs:
        inds.append(sample[-2])
        if len(sample[-1].shape) > 1:
            labels.append(torch.where(sample[-1][:, 1] == 1)[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
        torch.tensor(inds).view(len(inputs), 1), torch.tensor(
            labels).view(len(inputs))
