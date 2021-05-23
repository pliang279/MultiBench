import os
import sys
import h5py
from typing import *
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from robustness.visual_robust import visual_robustness
from robustness.text_robust import text_robustness

import re
from vgg import VGGClassifier
import gensim.models.keyedvectors as word2vec


import json
from PIL import Image
from typing import *
import os
from tqdm import tqdm

import numpy as np


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


def get_dataloader(path:str,)->Tuple[Dict]:
    
    '''
    return: 
    {filename1:{
        "image": ndarray,
        "plot": List[str],
        "label": ndarray, (one hot)
    }
    filename2:{
        ...
    }
    }
    '''

    split_file = os.path.join(path, "split.json")
    with open(split_file, "r") as f:
        split = json.load(f)

    label_file = os.path.join(path, "labels.json")
    with open(label_file, "r") as f:
        label_map = json.load(f) 
    
    dataset = os.path.join(path, "dataset")

    # traindata = {}
    # for name in tqdm(split["train"]):
    #     traindata[name] = process_data(name, dataset, labels)
    # devdata = {}
    # for name in tqdm(split["dev"]):
    #     devdata[name] = process_data(name, dataset, labels)
    testdata = []
    images = []
    texts = []
    labels = []
    for name in tqdm(split["test"]):
        data = process_data(name, dataset, label_map)
        images.append(data['image'])
        texts.append(data['plot'][0])
        labels.append(data['label'])
    clsf = VGGClassifier(model_path='/home/pliang/multibench/MultiBench/datasets/imdb/vgg16.tar', synset_words='synset_words.txt')
    googleword2vec = word2vec.KeyedVectors.load_word2vec_format('/home/pliang/multibench/MultiBench/datasets/imdb/GoogleNews-vectors-negative300.bin.gz', binary=True)
    for noise_level in range(10):
        images_robust = visual_robustness(images, noise_level=noise_level/10)
        texts_robust = text_robustness(texts, noise_level=noise_level/10)
        vgg_features = []
        text_features = []
        
        for im in tqdm(images_robust):
            vgg_features.append(clsf.get_features(Image.fromarray(im)).reshape((-1,)))      
        for words in tqdm(texts_robust):
            text_features.append(np.array([googleword2vec[w] for w in words if w in googleword2vec]).mean(axis=0))
        testdata.append([(text_features[i], vgg_features[i], labels[i]) for i in range(len(text_features))])
    return testdata

