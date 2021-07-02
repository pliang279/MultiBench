import os
import sys
from typing import *
import pickle
import h5py
import numpy as np
from numpy.core.numeric import zeros_like

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
import torchtext as text
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from robustness.text_robust import text_robustness
from robustness.timeseries_robust import timeseries_robustness

np.seterr(divide='ignore', invalid='ignore')


def drop_entry(dataset):
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    # for ind, k in enumerate(dataset["vision"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # for ind, k in enumerate(dataset["audio"]):
    #     if k.sum() == 0:
    #         if ind not in drop:
    #             drop.append(ind)
    # print(drop)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


def get_rawtext(path, data_kind, vids):
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    for vid in vids:
        text = []
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = vid
        try:
            for word in f['words'][vid_id]['features']:
                if word[0] != b'sp':
                    text.append(word[0].decode('utf-8'))
            text_data.append(' '.join(text))
            new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


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
    # print('Vocab Length: {}'.format(len(tokens)))
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
        # TODO: fix some segs have more than 50 words (FIXed)
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings - len(d)):
                tmp.append(np.zeros(300, ))
            for x in d:
                tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:
        #     print(d)
        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


class Affectdataset(Dataset):

    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None) -> None:
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task

    def __getitem__(self, ind):

        # vision = torch.tensor(vision)
        # audio = torch.tensor(audio)
        # text = torch.tensor(text)

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])

        # print(vision.shape)
        # print(audio.shape)
        # print(text.shape)

        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
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

        # z-normalize data
        vision = torch.nan_to_num((vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
        audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
        text = torch.nan_to_num((text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        label = torch.tensor(self.dataset['labels'][ind]).long() if self.task == "classification" else torch.tensor(
            self.dataset['labels'][ind]).float()

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind, \
                    label]
        else:
            return [vision, audio, text, ind, label]

    def __len__(self):
        return self.dataset['vision'].shape[0]


def get_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, train_shuffle: bool = True,
        num_workers: int = 4, flatten_time_series: bool = False, task=None,
        raw_path='/home/pliang/multibench/affect/mosi/mosi.hdf5') -> DataLoader:
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])

    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]

    train = DataLoader(Affectdataset(processed_dataset['train'], flatten_time_series, task=task), \
                       shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)
    valid = DataLoader(Affectdataset(processed_dataset['valid'], flatten_time_series, task=task), \
                       shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)
    # test = DataLoader(Affectdataset(processed_dataset['test'], flatten_time_series, task=task), \
    #                   shuffle=False, num_workers=num_workers, batch_size=batch_size, \
    #                   collate_fn=process)

    vids = [id for id in alldata['test']['id']]

    file_type = raw_path.split('.')[-1]  # hdf5
    rawtext, vids = get_rawtext(raw_path, file_type, vids)

    # Add text noises
    robust_text = []
    robust_text_numpy = []
    for i in range(10):
        test = dict()
        test['vision'] = alldata['test']["vision"]
        test['audio'] = alldata['test']["audio"]
        test['text'] = glove_embeddings(text_robustness(rawtext, noise_level=i / 10), vids)
        test['labels'] = alldata['test']["labels"]
        test = drop_entry(test)

        robust_text_numpy.append(test['text'])

        robust_text.append(
            DataLoader(Affectdataset(test, flatten_time_series, task=task), shuffle=False, num_workers=num_workers,
                       batch_size=batch_size, collate_fn=process))

    # Add visual noises
    robust_vision = []
    for i in range(10):
        test = dict()
        test['vision'] = timeseries_robustness([alldata['test']['vision'].copy()], noise_level=i / 10, rand_drop=False)[
            0]
        # print('vision shape: {}'.format(test['vision'].shape))
        test['audio'] = alldata['test']["audio"].copy()
        test['text'] = alldata['test']['text'].copy()
        test['labels'] = alldata['test']["labels"]
        test = drop_entry(test)
        print('test entries: {}'.format(test['vision'].shape))

        robust_vision.append(
            DataLoader(Affectdataset(test, flatten_time_series, task=task), shuffle=False, num_workers=num_workers,
                       batch_size=batch_size, collate_fn=process))

    # Add audio noises
    robust_audio = []
    for i in range(10):
        test = dict()
        test['vision'] = alldata['test']["vision"].copy()
        test['audio'] = timeseries_robustness([alldata['test']['audio'].copy()], noise_level=i / 10, rand_drop=False)[0]
        test['text'] = alldata['test']['text'].copy()
        test['labels'] = alldata['test']["labels"]
        test = drop_entry(test)
        print('test entries: {}'.format(test['vision'].shape))

        robust_audio.append(
            DataLoader(Affectdataset(test, flatten_time_series, task=task), shuffle=False, num_workers=num_workers,
                       batch_size=batch_size, collate_fn=process))

    # Add timeseries noises

    # for i, text in enumerate(robust_text_numpy):
    #     print(text.shape)
    #     alldata_test = timeseries_robustness([alldata['test']['vision'], alldata['test']['audio'], text], noise_level=i/10)
    #     test.append(alldata_test)

    robust_timeseries = []
    # alldata['test'] = drop_entry(alldata['test'])
    for i in range(10):
        robust_timeseries_tmp = timeseries_robustness(
            [alldata['test']['vision'].copy(), alldata['test']['audio'].copy(), alldata['test']['text'].copy()],
            noise_level=i / 10, rand_drop=False)
        # print('shape: {}'.format(robust_timeseries_tmp[1].shape))
        test = dict()
        test['vision'] = robust_timeseries_tmp[0]
        test['audio'] = robust_timeseries_tmp[1]
        test['text'] = robust_timeseries_tmp[2]
        test['labels'] = alldata['test']['labels']
        test = drop_entry(test)
        print('test entries: {}'.format(test['vision'].shape))

        robust_timeseries.append(
            DataLoader(Affectdataset(test, flatten_time_series, task=task), shuffle=False, num_workers=num_workers,
                       batch_size=batch_size, collate_fn=process))
    test_robust_data = dict()
    test_robust_data['robust_text'] = robust_text
    test_robust_data['robust_vision'] = robust_vision
    test_robust_data['robust_audio'] = robust_audio
    test_robust_data['robust_timeseries'] = robust_timeseries
    return train, valid, test_robust_data


def process(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        processed_input.append(pad_sequence(feature, batch_first=True))

    for sample in inputs:
        # print(sample[-1].shape)
        inds.append(sample[-2])
        if len(sample[-1].shape) > 2:
            labels.append(torch.where(sample[-1][:, 1] == 1)[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
           torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)


if __name__ == '__main__':
    train, valid, test_robust = get_dataloader('humor.pkl', raw_path='/home/pliang/multibench/affect/mosei/mosei.hdf5')

    keys = list(test_robust.keys())
    print(keys)

    # test_robust[keys[0]][1]
    for batch in test_robust[keys[1]][0]:
        for b in batch[0]:
            print(b.shape)
        print(batch[1])
        print(batch[2])
        break


