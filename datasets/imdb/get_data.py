"""Implements dataloaders for IMDB dataset."""

from tqdm import tqdm
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import h5py
from gensim.models import KeyedVectors
from .vgg import VGGClassifier
from robustness.text_robust import add_text_noise
from robustness.visual_robust import add_visual_noise
import os
import sys
from typing import *
import numpy as np

sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')


class IMDBDataset(Dataset):
    """Implements a torch Dataset class for the imdb dataset."""
    
    def __init__(self, file: h5py.File, start_ind: int, end_ind: int, vggfeature: bool = False) -> None:
        """Initialize IMDBDataset object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.file = file
        self.start_ind = start_ind
        self.size = end_ind-start_ind
        self.vggfeature = vggfeature

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.file, 'r')
        text = self.dataset["features"][ind+self.start_ind]
        image = self.dataset["images"][ind+self.start_ind] if not self.vggfeature else \
            self.dataset["vgg_features"][ind+self.start_ind]
        label = self.dataset["genres"][ind+self.start_ind]

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size


class IMDBDataset_robust(Dataset):
    """Implements a torch Dataset class for the imdb dataset that uses robustness measures as data augmentation."""

    def __init__(self, dataset, start_ind: int, end_ind: int) -> None:
        """Initialize IMDBDataset_robust object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.dataset = dataset
        self.start_ind = start_ind
        self.size = end_ind-start_ind

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        text = self.dataset[ind+self.start_ind][0]
        image = self.dataset[ind+self.start_ind][1]
        label = self.dataset[ind+self.start_ind][2]

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size


def _process_data(filename, path):
    data = {}
    filepath = os.path.join(path, filename)

    with Image.open(filepath+".jpeg") as f:
        image = np.array(f.convert("RGB"))
        data["image"] = image

    with open(filepath+".json", "r") as f:
        info = json.load(f)

        plot = info["plot"]
        data["plot"] = plot

    return data


def get_dataloader(path: str, test_path: str, num_workers: int = 8, train_shuffle: bool = True, batch_size: int = 40, vgg: bool = False, skip_process=False, no_robust=False) -> Tuple[Dict]:
    """Get dataloaders for IMDB dataset.

    Args:
        path (str): Path to training datafile.
        test_path (str): Path to test datafile.
        num_workers (int, optional): Number of workers to load data in. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        batch_size (int, optional): Batch size of data. Defaults to 40.
        vgg (bool, optional): Whether to return raw images or pre-processed vgg features. Defaults to False.
        skip_process (bool, optional): Whether to pre-process data or not. Defaults to False.
        no_robust (bool, optional): Whether to not use robustness measures as augmentation. Defaults to False.

    Returns:
        Tuple[Dict]: Tuple of Training dataloader, Validation dataloader, Test Dataloader
    """
    train_dataloader = DataLoader(IMDBDataset(path, 0, 15552, vgg),
                                  shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(IMDBDataset(path, 15552, 18160, vgg),
                                shuffle=False, num_workers=num_workers, batch_size=batch_size)
    if no_robust:
        test_dataloader = DataLoader(IMDBDataset(path, 18160, 25959, vgg),
                                     shuffle=False, num_workers=num_workers, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader

    test_dataset = h5py.File(path, 'r')
    test_text = test_dataset['features'][18160:25959]
    test_vision = test_dataset['vgg_features'][18160:25959]
    labels = test_dataset["genres"][18160:25959]
    names = test_dataset["imdb_ids"][18160:25959]

    dataset = os.path.join(test_path, "dataset")

    if not skip_process:
        clsf = VGGClassifier(
            model_path='/home/pliang/multibench/MultiBench/datasets/imdb/vgg16.tar', synset_words='synset_words.txt')
        googleword2vec = KeyedVectors.load_word2vec_format(
            '/home/pliang/multibench/MultiBench/datasets/imdb/GoogleNews-vectors-negative300.bin.gz', binary=True)

        images = []
        texts = []
        for name in tqdm(names):
            name = name.decode("utf-8")
            data = _process_data(name, dataset)
            images.append(data['image'])
            plot_id = np.array([len(p) for p in data['plot']]).argmax()
            texts.append(data['plot'][plot_id])

    # Add visual noises
    robust_vision = []
    for noise_level in range(11):
        vgg_filename = os.path.join(
            os.getcwd(), 'vgg_features_{}.npy'.format(noise_level))
        if not skip_process:
            vgg_features = []
            images_robust = add_visual_noise(
                images, noise_level=noise_level/10)
            for im in tqdm(images_robust):
                vgg_features.append(clsf.get_features(
                    Image.fromarray(im)).reshape((-1,)))
            np.save(vgg_filename, vgg_features)
        else:
            assert os.path.exists(vgg_filename) == True
            vgg_features = np.load(vgg_filename, allow_pickle=True)
        robust_vision.append([(test_text[i], vgg_features[i], labels[i])
                             for i in range(len(vgg_features))])

    test_dataloader = dict()
    test_dataloader['image'] = []
    for test in robust_vision:
        test_dataloader['image'].append(DataLoader(IMDBDataset_robust(test, 0, len(
            test)), shuffle=False, num_workers=num_workers, batch_size=batch_size))

    # Add text noises
    robust_text = []
    for noise_level in range(11):
        text_filename = os.path.join(
            os.getcwd(), 'text_features_{}.npy'.format(noise_level))
        if not skip_process:
            text_features = []
            texts_robust = add_text_noise(texts, noise_level=noise_level/10)
            for words in tqdm(texts_robust):
                words = words.split()
                if len([googleword2vec[w] for w in words if w in googleword2vec]) == 0:
                    text_features.append(np.zeros((300,)))
                else:
                    text_features.append(np.array(
                        [googleword2vec[w] for w in words if w in googleword2vec]).mean(axis=0))
            np.save(text_filename, text_features)
        else:
            assert os.path.exists(text_filename) == True
            text_features = np.load(text_filename, allow_pickle=True)
        robust_text.append([(text_features[i], test_vision[i], labels[i])
                           for i in range(len(text_features))])
    test_dataloader['text'] = []
    for test in robust_text:
        test_dataloader['text'].append(DataLoader(IMDBDataset_robust(test, 0, len(
            test)), shuffle=False, num_workers=num_workers, batch_size=batch_size))
    return train_dataloader, val_dataloader, test_dataloader
