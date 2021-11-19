from mosi_split import train_fold, valid_fold, test_fold
import pickle
import sys
import os
import numpy as np
import h5py
import re
import torchtext as text
from collections import defaultdict
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


def lpad(this_array, seq_len):
    temp_array = np.concatenate([np.zeros(
        [seq_len]+list(this_array.shape[1:])), this_array], axis=0)[-seq_len:, ...]
    return temp_array


def detect_entry_fold(entry, folds):
    entry_id = entry.split("[")[0]
    for i in range(len(folds)):
        if entry_id in folds[i]:
            return i
    return None


folds = [train_fold, valid_fold, test_fold]
print('folds:')
print(len(train_fold))
print(len(valid_fold))
print(len(test_fold))

affect_data = h5py.File('/home/pliang/multibench/affect/mosi/mosi.hdf5', 'r')
print(affect_data.keys())

AUDIO = 'COVAREP'
VIDEO = 'FACET_4.2'
WORD = 'words'
labels = ['Opinion Segment Labels']

csds = [AUDIO, VIDEO, labels[0]]

seq_len = 50

keys = list(affect_data[WORD].keys())
print(len(keys))


def get_rawtext(path, data_kind, vids):
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
        text_data = []
        new_vids = []
        count = 0
        for vid in vids:
            text = []
            # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
            # vid_id = '{}[{}]'.format(id, seg)
            vid_id = vid
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
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings-len(d)):
                tmp.append(np.zeros(300,))
            for x in d:
                tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:
        
        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


def get_audio_visual_text(csds, seq_len, text_data, vids):
    data = [{} for _ in range(3)]
    output = [{} for _ in range(3)]

    for i in range(len(folds)):
        for csd in csds:
            data[i][csd] = []
        data[i]['words'] = []
        data[i]['id'] = []

    for i, key in enumerate(vids):
        which_fold = detect_entry_fold(key, folds)
        
        if which_fold == None:
            print("Key %s doesn't belong to any fold ... " %
                  str(key), error=False)
            continue
        for csd in csds:
            this_array = affect_data[csd][key]["features"]
            if csd in labels:
                data[which_fold][csd].append(this_array)
            else:
                data[which_fold][csd].append(lpad(this_array, seq_len=seq_len))
        data[which_fold]['words'].append(text_data[i])
        data[which_fold]['id'].append(key)

    for i in range(len(folds)):
        for csd in csds:
            output[i][csd] = np.array(data[i][csd])
        output[i]['words'] = np.stack(data[i]['words'])
        output[i]['id'] = data[i]['id']

    fold_names = ["train", "valid", "test"]
    for i in range(3):
        for csd in csds:
            print("Shape of the %s computational sequence for %s fold is %s" %
                  (csd, fold_names[i], output[i][csd].shape))
        print("Shape of the %s computational sequence for %s fold is %s" %
              ('words', fold_names[i], output[i]['words'].shape))
    return output


raw_text, vids = get_rawtext(
    '/home/pliang/multibench/affect/mosi/mosi.hdf5', 'hdf5', keys)
print(raw_text[0])
print(vids[0])
text_glove = glove_embeddings(raw_text, vids)
print(text_glove.shape)

audio_video_text = get_audio_visual_text(
    csds, seq_len=seq_len, text_data=text_glove, vids=vids)
print(len(audio_video_text))
print(audio_video_text[0].keys())

all_data = {}
fold_names = ["train", "valid", "test"]
key_sets = ['audio', 'vision', 'text', 'labels', 'id']

for i, fold in enumerate(fold_names):
    all_data[fold] = {}
    all_data[fold]['vision'] = audio_video_text[i][VIDEO]
    all_data[fold]['audio'] = audio_video_text[i][AUDIO]
    all_data[fold]['text'] = audio_video_text[i]['words']
    all_data[fold]['labels'] = audio_video_text[i][labels[0]]
    all_data[fold]['id'] = audio_video_text[i]['id']


with open('mosi_raw.pkl', 'wb') as f:
    pickle.dump(all_data, f)
