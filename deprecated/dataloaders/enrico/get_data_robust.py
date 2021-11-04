from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import csv
import os
import json
from collections import Counter

import torch
from torchvision import transforms

from PIL import Image
import numpy as np
from robustness.visual_robust import add_visual_noise

# helper function for extracting UI elements from hierarchy


def add_screen_elements(tree, element_list):
    if 'children' in tree and len(tree['children']) > 0:
        # we are at an intermediate node
        for child in tree['children']:
            add_screen_elements(child, element_list)
    else:
        # we are at a leaf node
        if 'bounds' in tree and 'componentLabel' in tree:
            # valid leaf node
            nodeBounds = tree['bounds']
            nodeLabel = tree['componentLabel']
            node = (nodeBounds, nodeLabel)
            element_list.append(node)


class EnricoDataset(Dataset):
    def __init__(self, data_dir, noise_level, img_noise=False, wireframe_noise=False, img_dim_x=128, img_dim_y=256, random_seed=42, train_split=0.65, val_split=0.15, test_split=0.2, normalize_image=False, seq_len=64):
        super(EnricoDataset, self).__init__()
        self.noise_level = noise_level
        self.img_noise = img_noise
        self.wireframe_noise = wireframe_noise
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.seq_len = seq_len
        csv_file = os.path.join(data_dir, "design_topics.csv")
        self.img_dir = os.path.join(data_dir, "screenshots")
        self.wireframe_dir = os.path.join(data_dir, "wireframes")
        self.hierarchy_dir = os.path.join(data_dir, "hierarchies")
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            example_list = list(reader)

        # the wireframe files are corrupted for these files
        IGNORES = set(["50105", "50109"])
        example_list = [
            e for e in example_list if e['screen_id'] not in IGNORES]

        self.example_list = example_list

        keys = list(range(len(example_list)))
        # shuffle and create splits
        random.Random(random_seed).shuffle(keys)

        # test split is at the end
        start_index = int(len(example_list) * (train_split + val_split))
        stop_index = len(example_list)

        # only keep examples in the current split
        keys = keys[start_index:stop_index]
        self.keys = keys

        img_transforms = [
            transforms.Resize((img_dim_y, img_dim_x)),
            transforms.ToTensor()
        ]
        if normalize_image:
            img_transforms.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # pytorch image transforms
        self.img_transforms = transforms.Compose(img_transforms)

        # make maps
        topics = set()
        for e in example_list:
            topics.add(e['topic'])
        topics = sorted(list(topics))

        idx2Topic = {}
        topic2Idx = {}

        for i in range(len(topics)):
            idx2Topic[i] = topics[i]
            topic2Idx[topics[i]] = i

        self.idx2Topic = idx2Topic
        self.topic2Idx = topic2Idx

        UI_TYPES = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab",
                    "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

        idx2Label = {}
        label2Idx = {}

        for i in range(len(UI_TYPES)):
            idx2Label[i] = UI_TYPES[i]
            label2Idx[UI_TYPES[i]] = i

        self.idx2Label = idx2Label
        self.label2Idx = label2Idx
        self.ui_types = UI_TYPES

    def __len__(self):
        return len(self.keys)

    def featurizeElement(self, element):
        bounds, label = element
        labelOneHot = [0 for _ in range(len(self.ui_types))]
        labelOneHot[self.label2Idx[label]] = 1
        return bounds, labelOneHot

    def __getitem__(self, idx):
        example = self.example_list[self.keys[idx]]
        screenId = example['screen_id']
        # image modality
        screenImg = Image.open(os.path.join(
            self.img_dir, screenId + ".jpg")).convert("RGB")
        if self.img_noise:
            screenImg = Image.fromarray(add_visual_noise(
                [np.array(screenImg)], noise_level=self.noise_level)[0])
        screenImg = self.img_transforms(screenImg)
        # wireframe image modality
        screenWireframeImg = Image.open(os.path.join(
            self.wireframe_dir, screenId + ".png")).convert("RGB")
        if self.wireframe_noise:
            screenWireframeImg = Image.fromarray(add_visual_noise(
                [np.array(screenWireframeImg)], noise_level=self.noise_level)[0])
        screenWireframeImg = self.img_transforms(screenWireframeImg)
        # label
        screenLabel = self.topic2Idx[example['topic']]
        return [screenImg, screenWireframeImg, screenLabel]


def get_dataloader_robust(data_dir, batch_size=32, num_workers=0, img_noise=True, wireframe_noise=True):
    ds_test_img = []
    ds_test_wireframe = []
    dl_test = []
    if img_noise:
        for i in range(11):
            ds_test_img.append(EnricoDataset(
                data_dir, img_noise=True, noise_level=i/10))
        dl_test.append([DataLoader(test, shuffle=False, num_workers=num_workers,
                       batch_size=batch_size) for test in ds_test_img])
    if wireframe_noise:
        for i in range(11):
            ds_test_wireframe.append(EnricoDataset(
                data_dir, wireframe_noise=True, noise_level=i/10))
        dl_test.append([DataLoader(test, shuffle=False, num_workers=num_workers,
                       batch_size=batch_size) for test in ds_test_wireframe])
    return dl_test
