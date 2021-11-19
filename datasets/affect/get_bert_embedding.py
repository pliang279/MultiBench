import torch
from torch import nn
from transformers import AutoTokenizer, pipeline, BertModel
import h5py
import pickle
import numpy as np


model_name = "bert-base-uncased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
# features_extractor = pipeline('feature-extraction', model=model_name, tokenizer=model_name)
bert = BertModel.from_pretrained(model_name)
bert.config.output_hidden_states = True

# use pipline to extract all the features, (num_points, max_seq_length, feature_dim): np.ndarray
# contextual embedding:if True output the last hidden state of bert, if False, output the embedding of words
def get_bert_features(all_text, contextual_embedding=False, batch_size=500, max_len=None):
    output_bert_features = []
    if max_len == None:
        max_len = max([len([ms for ms in s.split() if len(ms) > 0]) for s in all_text])
    print(max_len)
    print(len(all_text))

    for i in range(0, len(all_text), batch_size):
        
        inputs = tokenizer(all_text[i: i+batch_size], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

        bert_feartures = bert(**inputs)

        outputs = bert_feartures.hidden_states
        if contextual_embedding:
            output_bert_features.append(outputs[-1].detach().numpy())
        else:
            output_bert_features.append(outputs[0].detach().numpy())
            print(outputs[0].detach().numpy().shape)
        print('i = {} finished!'.format(i))
    
    print(np.concatenate(output_bert_features).shape)
    return np.concatenate(output_bert_features)

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
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
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
        if '##' in t:  # deal with BERT sub words
            id_list.append(idx)
        elif '\'' == t:
            id_list.append(idx)
            if i+1 < len(tokenized_text):  # deal with [she's] [you're] [you'll] etc. or [sisters' parents] [brothers']
                if ''.join([tokenized_text[i-1], t, tokenized_text[i+1]]) in orig_text or tokenized_text[i+1] == 's':
                    idx -= 1
        elif '-' == t:  # deal with e.g. [good-time]
            id_list.append(idx)
            idx -= 1
        elif '{' == t:  # deal with {lg} and {cg} marks
            id_list.append(idx+1)
        elif '}' == t:
            id_list.append(idx)
        else:
            idx += 1
            id_list.append(idx)
    if len(id_list) > 0:
        ori_list = [k.strip() for k in orig_text.split(' ') if len(k) > 0]
        if len(ori_list) != id_list[-1]+1:
            print(orig_text)
            print(tokenized_text)
            print(id_list)
    return id_list


def bert_version_data(data, raw_path, keys, max_padding=50, bert_max_len=None):

    file_type = raw_path.split('.')[-1]
    sarcasm_text, _ = get_rawtext(raw_path, file_type, keys)

    bert_features = get_bert_features(sarcasm_text, contextual_embedding=False, max_len=bert_max_len)  # (N, MAX_LEN, 768) for sarcasm
    
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

    new_bert_features = []
    if bert_features.shape[1] >= max_padding:
        for b in bert_features:
            new_bert_features.append(b[:max_padding, :])
    else:
        for b in bert_features:
            new_bert_features.append(np.pad(b, ((0, max_padding-bert_features.shape[1]), (0, 0))))
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
    
    
