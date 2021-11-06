import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from .utils import augment_val

from datasets.robotics import ProcessForce, ToTensor
from datasets.robotics import MultimodalManipulationDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


def combine_modalitiesbuilder(unimodal, output):
    def combine_modalities(data):
        if unimodal == "force":
            return [data['force'], data['action'], data[output]]
        if unimodal == "proprio":
            return [data['proprio'], data['action'], data[output]]
        if unimodal == "image":
            return [data['image'], data['depth'].transpose(0, 2).transpose(1, 2), data['action'], data[output]]
        return [
            data['image'],
            data['force'],
            data['proprio'],
            data['depth'].transpose(0, 2).transpose(1, 2),
            data['action'],
            data[output],
        ]
    return combine_modalities


def get_data(device, configs, filedirprefix="", unimodal=None, output='contact_next'):
    filename_list = []
    for file in os.listdir(configs['dataset']):
        if file.endswith(".h5"):
            filename_list.append(configs['dataset'] + file)

    print(
        "Number of files in multifile dataset = {}".format(len(filename_list))
    )

    val_filename_list = []

    val_index = np.random.randint(
        0, len(filename_list), int(len(filename_list) * configs['val_ratio'])
    )

    for index in val_index:
        val_filename_list.append(filename_list[index])

    while val_index.size > 0:
        filename_list.pop(val_index[0])
        val_index = np.where(
            val_index > val_index[0], val_index - 1, val_index)
        val_index = val_index[1:]

    print("Initial finished")

    val_filename_list1, filename_list1 = augment_val(
        val_filename_list, filename_list
    )

    print("Listing finished")

    dataloaders = {}
    samplers = {}
    datasets = {}

    samplers["val"] = SubsetRandomSampler(
        range(len(val_filename_list1) * (configs['ep_length'] - 1))
    )
    samplers["train"] = SubsetRandomSampler(
        range(len(filename_list1) * (configs['ep_length'] - 1))
    )

    print("Sampler finished")

    datasets["train"] = MultimodalManipulationDataset(
        filename_list1,
        transform=transforms.Compose(
            [
                ProcessForce(32, "force", tanh=True),
                ProcessForce(32, "unpaired_force", tanh=True),
                ToTensor(device=device),
                combine_modalitiesbuilder(unimodal, output),
            ]
        ),
        episode_length=configs['ep_length'],
        training_type=configs['training_type'],
        action_dim=configs['action_dim'],
        filedirprefix=filedirprefix
    )

    datasets["val"] = MultimodalManipulationDataset(
        val_filename_list1,
        transform=transforms.Compose(
            [
                ProcessForce(32, "force", tanh=True),
                ProcessForce(32, "unpaired_force", tanh=True),
                ToTensor(device=device),
                combine_modalitiesbuilder(unimodal, output),
            ]
        ),
        episode_length=configs['ep_length'],
        training_type=configs['training_type'],
        action_dim=configs['action_dim'],

    )

    print("Dataset finished")

    dataloaders["val"] = DataLoader(
        datasets["val"],
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers'],
        sampler=samplers["val"],
        pin_memory=True,
        drop_last=True,
    )
    dataloaders["train"] = DataLoader(
        datasets["train"],
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers'],
        sampler=samplers["train"],
        pin_memory=True,
        drop_last=True,
    )

    print("Finished setting up date")
    return dataloaders['train'], dataloaders['val']
