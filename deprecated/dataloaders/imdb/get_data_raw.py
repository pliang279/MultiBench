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
            if label in labels:
                genre[labels[label]] = 1
        data["label"] = genre

    return data


def get_dataloader(path: str,) -> Tuple[Dict]:
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
        labels = json.load(f)

    dataset = os.path.join(path, "dataset")

    traindata = {}
    for name in tqdm(split["train"]):
        traindata[name] = process_data(name, dataset, labels)
    devdata = {}
    for name in tqdm(split["dev"]):
        devdata[name] = process_data(name, dataset, labels)
    testdata = {}
    for name in tqdm(split["test"]):
        testdata[name] = process_data(name, dataset, labels)

    return traindata, devdata, testdata
