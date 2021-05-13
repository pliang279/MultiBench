from torch.utils.data import Dataset, DataLoader
import random
import csv
import os
import json

import torch
from torchvision import transforms

from PIL import Image

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
    def __init__(self, data_dir, mode="train", img_dim=224, random_seed=42, train_split=0.7, val_split=0.15, test_split=0.15, normalize_image=True, seq_len=64):
        super(EnricoDataset, self).__init__()
        self.img_dim = img_dim
        self.seq_len = seq_len
        csv_file = os.path.join(data_dir, "design_topics.csv")
        self.img_dir = os.path.join(data_dir, "screenshots")
        self.hierarchy_dir = os.path.join(data_dir, "hierarchies")
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            example_list = list(reader)

        self.example_list = example_list

        keys = list(range(len(example_list)))
        # shuffle and create splits
        random.Random(random_seed).shuffle(keys)
        
        if mode == "train":
            # train split is at the front
            start_index = 0
            stop_index = int(len(example_list) * train_split)
        elif mode == "val":
            # val split is in the middle
            start_index = int(len(example_list) * train_split)
            stop_index = int(len(example_list) * (train_split + val_split))
        elif mode == "test":
            # test split is at the end
            start_index = int(len(example_list) * (train_split + val_split))
            stop_index = len(example_list)

        # only keep examples in the current split
        keys = keys[start_index:stop_index]
        self.keys = keys

        img_transforms = [
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor()
        ]
        if normalize_image:
            img_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

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

        UI_TYPES = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

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
        screenImg = Image.open(os.path.join(self.img_dir, screenId + ".jpg"))
        screenImg = self.img_transforms(screenImg)
        # wireframe modalities
        treePath = os.path.join(self.hierarchy_dir, screenId + ".json")
        with open(treePath, "r", errors="ignore", encoding="utf-8") as f:
            tree = json.load(f)
        treeElements = []
        add_screen_elements(tree, treeElements)
        random.shuffle(treeElements)
        elements = []
        labels = []
        for e in treeElements:
            b, l = self.featurizeElement(e)
            elements.append(torch.tensor(b))
            labels.append(torch.tensor(l))
        
        screenWireframeBoundsPadded = torch.zeros(self.seq_len, 4)
        screenWireframeLabelsPadded = torch.zeros(self.seq_len, len(self.ui_types))
        if len(elements) > 0:
            screenWireframeBounds = torch.stack(elements, dim=0)
            padLen = min(self.seq_len, screenWireframeBounds.shape[0])
            screenWireframeBoundsPadded[:padLen, :] = screenWireframeBounds[:padLen, :]
            screenWireframeLabels = torch.stack(labels, dim=0)
            screenWireframeLabelsPadded[:padLen, :] = screenWireframeLabels[:padLen, :]
        # label
        screenLabel = self.topic2Idx[example['topic']]
        # return a list where each index is a modality
        # return [screenImg, (screenWireframeBoundsPadded, padLen), (screenWireframeLabelsPadded, padLen), screenLabel]
        return [screenImg, screenWireframeBoundsPadded, screenWireframeLabelsPadded, screenLabel]

def get_dataloader(data_dir, batch_size=8, num_workers=0, train_shuffle=True, normalize_image=True):
    ds_train = EnricoDataset(data_dir, mode="train")
    ds_val = EnricoDataset(data_dir, mode="val")
    ds_test = EnricoDataset(data_dir, mode="test")

    dl_train = DataLoader(ds_train, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    dl_val = DataLoader(ds_val, shuffle=False, num_workers=num_workers, batch_size=batch_size)
    dl_test = DataLoader(ds_test, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return dl_train, dl_val, dl_test