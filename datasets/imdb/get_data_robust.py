import os
import sys
from typing import *
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from robustness.visual_robust import visual_robustness
from robustness.text_robust import text_robustness

import re
from vgg import VGGClassifier
import gensim.models.keyedvectors as word2vec
import torch
import h5py
from typing import *
from torch.utils.data import Dataset, DataLoader

import json
from PIL import Image
from typing import *
import os
from tqdm import tqdm



class IMDBDataset(Dataset):
    
    def __init__(self, file:h5py.File, start_ind:int, end_ind:int, vggfeature:bool=False) -> None:
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

        return text, image, label

    def __len__(self):
        return self.size


class IMDBDataset_robust(Dataset):
    
    def __init__(self, dataset, start_ind:int, end_ind:int) -> None:
        self.dataset = dataset
        self.start_ind = start_ind
        self.size = end_ind-start_ind

    def __getitem__(self, ind):
        text = self.dataset[ind+self.start_ind][0]
        image = self.dataset[ind+self.start_ind][1]
        label = self.dataset[ind+self.start_ind][2]

        return text, image, label

    def __len__(self):
        return self.size

def process_data(filename, path, labels):
    data = {}
    filepath = os.path.join(path, filename)

    with Image.open(filepath+".jpeg") as f:
        image = np.array(f.convert("RGB"))
        data["image"] = image
    
    with open(filepath+".json", "r") as f:
        info = json.load(f)
        
        plot = info["plot"]
        data["plot"] = plot
        
        genre = np.zeros(len(labels))
        for label in info["genres"]:
            if label in labels:
                genre[labels[label]] = 1
        data["label"] = genre

    return data


def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()


def get_dataloader(
    path:str, num_workers:int=8, train_shuffle:bool=True, batch_size:int=40, vgg:bool=False)->Tuple[DataLoader]:
    
    train_dataloader = DataLoader(IMDBDataset(path, 0, 15552, vgg), \
        shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(IMDBDataset(path, 15552, 18160, vgg), \
        shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return train_dataloader, val_dataloader


def get_dataloader_robust(path:str,num_workers:int=8, train_shuffle:bool=True, batch_size:int=40)->Tuple[Dict]:

    split_file = os.path.join(path, "split.json")
    with open(split_file, "r") as f:
        split = json.load(f)

    label_file = os.path.join(path, "labels.json")
    with open(label_file, "r") as f:
        label_map = json.load(f) 
    
    dataset = os.path.join(path, "dataset")

    testdata = []
    # clsf = VGGClassifier(model_path='/home/pliang/multibench/MultiBench/datasets/imdb/vgg16.tar', synset_words='synset_words.txt')
    # googleword2vec = word2vec.KeyedVectors.load_word2vec_format('/home/pliang/multibench/MultiBench/datasets/imdb/GoogleNews-vectors-negative300.bin.gz', binary=True)
    images = []
    texts = []
    labels = []
    for name in tqdm(split["test"]):
        data = process_data(name, dataset, label_map)
        images.append(data['image'])
        texts.append(data['plot'][0])
        labels.append(data['label'])
    for noise_level in range(11):
        vgg_filename = os.path.join(os.getcwd(), 'vgg_features_{}.npy'.format(noise_level))
        text_filename = os.path.join(os.getcwd(), 'text_features_{}.npy'.format(noise_level))
        # extract_vgg = not os.path.exists(vgg_filename)
        # extract_text = not os.path.exists(text_filename)
        # vgg_features = []
        # text_features = []
        # if extract_vgg:
        #     images_robust = visual_robustness(images, noise_level=noise_level/10)
        #     for im in tqdm(images_robust):
        #         vgg_features.append(clsf.get_features(Image.fromarray(im)).reshape((-1,)))
        #     np.save(vgg_filename, vgg_features)
        # else:
        vgg_features = np.load(vgg_filename, allow_pickle=True)
        # if extract_text:
        #     texts_robust = text_robustness(texts, noise_level=noise_level/10)    
        #     for words in tqdm(texts_robust):
        #         if len([googleword2vec[w] for w in words if w in googleword2vec]) == 0:
        #             text_features.append(np.zeros((300,)))
        #         else:
        #             text_features.append(np.array([googleword2vec[w] for w in words if w in googleword2vec]).mean(axis=0))
        #     np.save(text_filename, text_features)
        # else:
        text_features = np.load(text_filename, allow_pickle=True)
        testdata.append([(text_features[i], vgg_features[i], labels[i]) for i in range(len(vgg_features))])
    test_dataloader = []
    for test in testdata:
        test_dataloader.append(DataLoader(IMDBDataset_robust(test, 0, len(test)), shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size))
    return test_dataloader

