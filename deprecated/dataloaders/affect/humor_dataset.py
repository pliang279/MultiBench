import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn


def load_pickle(pickle_file):
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


class HumorDataset(Dataset):

    def __init__(self, id_list, max_context_len=5, max_sen_len=20):
        self.id_list = id_list
        openface_file = "openface_features_sdk.pkl"
        covarep_file = "covarep_features_sdk.pkl"
        language_file = "language_sdk.pkl"
        word_embedding_list_file = "word_embedding_list.pkl"
        humor_label_file = "humor_label_sdk.pkl"

        self.word_aligned_openface_sdk = load_pickle(openface_file)
        self.word_aligned_covarep_sdk = load_pickle(covarep_file)
        self.language_sdk = load_pickle(language_file)
        self.word_embedding_list_sdk = load_pickle(word_embedding_list_file)
        self.humor_label_sdk = load_pickle(humor_label_file)
        self.of_d = 371
        self.cvp_d = 81
        self.glove_d = 300
        self.total_dim = self.glove_d + self.of_d + self.cvp_d
        self.max_context_len = max_context_len
        self.max_sen_len = max_sen_len

    # left padding with zero  vector upto maximum number of words in a sentence * glove embedding dimension
    def paded_word_idx(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        pad_w = np.concatenate((np.zeros(max_sen_len - len(seq)), seq), axis=0)
        pad_w = np.array([self.word_embedding_list_sdk[int(w_id)]
                         for w_id in pad_w])
        return pad_w

    # left padding with zero  vector upto maximum number of words in a sentence * covarep dimension
    def padded_covarep_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((np.zeros((max_sen_len - len(seq), self.cvp_d)), seq), axis=0)

        # left padding with zero  vector upto maximum number of words in a sentence * openface dimension
    def padded_openface_features(self, seq, max_sen_len=20, left_pad=1):
        seq = seq[0:max_sen_len]
        return np.concatenate((np.zeros(((max_sen_len - len(seq)), self.of_d)), seq), axis=0)

        # left padding with zero vectors upto maximum number of sentences in context * maximum num of words in a sentence * 456
    def padded_context_features(self, context_w, context_of, context_cvp, max_context_len=5, max_sen_len=20):
        context_w = context_w[-max_context_len:]
        context_of = context_of[-max_context_len:]
        context_cvp = context_cvp[-max_context_len:]

        padded_context = []
        for i in range(len(context_w)):
            p_seq_w = self.paded_word_idx(context_w[i], max_sen_len)
            p_seq_cvp = self.padded_covarep_features(
                context_cvp[i], max_sen_len)
            p_seq_of = self.padded_openface_features(
                context_of[i], max_sen_len)
            padded_context.append(np.concatenate(
                (p_seq_w, p_seq_cvp, p_seq_of), axis=1))

        pad_c_len = max_context_len - len(padded_context)
        padded_context = np.array(padded_context)

        # if there is no context
        if not padded_context.any():
            return np.zeros((max_context_len, max_sen_len, self.total_dim))

        return np.concatenate((np.zeros((pad_c_len, max_sen_len, self.total_dim)), padded_context), axis=0)

    def padded_punchline_features(self, punchline_w, punchline_of, punchline_cvp, max_sen_len=20, left_pad=1):

        p_seq_w = self.paded_word_idx(punchline_w, max_sen_len)
        p_seq_cvp = self.padded_covarep_features(punchline_cvp, max_sen_len)
        p_seq_of = self.padded_openface_features(punchline_of, max_sen_len)
        return np.concatenate((p_seq_w, p_seq_cvp, p_seq_of), axis=1)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        hid = self.id_list[index]
        punchline_w = np.array(
            self.language_sdk[hid]['punchline_embedding_indexes'])
        punchline_of = np.array(
            self.word_aligned_openface_sdk[hid]['punchline_features'])
        punchline_cvp = np.array(
            self.word_aligned_covarep_sdk[hid]['punchline_features'])

        context_w = np.array(
            self.language_sdk[hid]['context_embedding_indexes'])
        context_of = np.array(
            self.word_aligned_openface_sdk[hid]['context_features'])
        context_cvp = np.array(
            self.word_aligned_covarep_sdk[hid]['context_features'])

        # punchline feature
        x_p = torch.LongTensor(
            self.padded_punchline_features(punchline_w, punchline_of, punchline_cvp, self.max_sen_len))
        # context feature
        x_c = torch.LongTensor(
            self.padded_context_features(context_w, context_of, context_cvp, self.max_context_len, self.max_sen_len))

        y = torch.FloatTensor([self.humor_label_sdk[hid]])

        return x_p, x_c, y
