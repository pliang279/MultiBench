import torch
from torch import nn
from transformers import AutoTokenizer, pipeline
import h5py
import pickle
import numpy  as np


model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
features_extractor = pipeline('feature-extraction', model=model_name, tokenizer=model_name)


# use pipline to extract all the features, (num_points, max_seq_length, feature_dim): np.ndarray
def get_bert_features(bert_extractor, all_text):
    bert_feartures = bert_extractor(all_text)
    return np.array(bert_feartures)


# get raw text from the datasets
def get_rawtext(path, data_kind, vids=None):
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    if vids == None:
        vids = list(f.keys())

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


# cut the id lists with the max length, but didnt do padding here
# add the first one as [CLS] and the last one for [SEP]
def max_seq_len(id_list, max_len=50):
    new_id_list = []
    for id in id_list:
        if len(id) > 0:
            id.append(id[-1])  # [SEP]
            id.insert(0, id[0]) # [CLS]
        new_id_list.append(id[:max_len])
    return new_id_list


# since tokenizer splits the word into parts e.g. '##ing' or 'you're' -> 'you', ''', 're'
# we should get the corresponding ids for other modalities' features
# applied to modalities which aligned to words
def corresponding_other_modality_ids(orig_text, tokenized_text):
    id_list = []
    idx = -1
    for i, t in enumerate(tokenized_text):
        if '##' in t:
            id_list.append(idx)
        elif '\'' == t:
            id_list.append(idx)
            if len(tokenized_text[i+1]) <= 3:
                idx -= 1
        else:
            idx += 1
            id_list.append(idx)
    if len(id_list) > 0:
        ori_list = [k.strip() for k in orig_text.split(' ') if len(k) > 0]
        if len(ori_list) != id_list[-1]+1:
            print(orig_text)
            print(tokenized_text)
    return id_list


def bert_version_data(data, raw_path, keys, max_padding=50):

    file_type = raw_path.split('.')[-1]
    sarcasm_text, _ = get_rawtext(raw_path, file_type, keys)

    bert_features = get_bert_features(features_extractor, sarcasm_text)  # (690, 74, 768) for sarcasm
    
    # get corresponding ids
    other_modality_ids = []
    for origi_text in sarcasm_text:
        tokenized_sequence = tokenizer.tokenize(origi_text)
        other_modality_ids.append(corresponding_other_modality_ids(origi_text, tokenized_sequence))

    # apply max seq len, DON'T FORGET [CLS] and [SEP] token
    new_other_mids = max_seq_len(other_modality_ids, max_len=max_padding)

    # get other modal features and pad them to max len
    new_vision = []
    for i, v in enumerate(data['vision']):
        tmp = v[new_other_mids[i]]
        tmp = np.pad(tmp, ((0, max_padding - tmp.shape[0]), (0, 0)))
        new_vision.append(tmp)
    new_vision = np.stack(new_vision)

    new_audio = []
    for i, a in enumerate(data['audio']):
        tmp = a[new_other_mids[i]]
        tmp = np.pad(tmp, ((0, max_padding - tmp.shape[0]), (0, 0)))
        new_audio.append(tmp)
    new_audio = np.stack(new_audio)

    assert bert_features.shape[1] >= max_padding

    new_bert_features = []
    for b in bert_features:
        new_bert_features.append(b[:max_padding, :])
    new_bert_features = np.stack(new_bert_features)

    return {'vision': new_vision, 'audio': new_audio, 'text': new_bert_features}


if __name__ == '__main__':

    with open('/home/pliang/multibench/affect/sarcasm.pkl', "rb") as f:
        alldata = pickle.load(f)

    train_keys = list(alldata['train']['id'])
    print(alldata['train']['vision'].shape)

    raw_path = '/home/pliang/multibench/affect/sarcasm_raw_text.pkl'

    new_train_data = bert_version_data(alldata['train'], raw_path, train_keys)

    print(new_train_data['vision'].shape)
    print(new_train_data['audio'].shape)
    print(new_train_data['text'].shape)

    # ori = ['so', 'how', 'she\'s', 'aad', 'it', 'go']
    # test = ['so', 'how', 'she', '\'', 's', 'aa',  '##d', 'it', 'go']
    # print(test)
    # print(corresponding_other_modality_ids(ori, test))
