"""Implements dataloaders for robotics data."""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from .utils import augment_val

from datasets.robotics import ProcessForce, ToTensor
from datasets.robotics import MultimodalManipulationDataset, MultimodalManipulationDataset_robust
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


def combine_modalitiesbuilder(unimodal, output):
    """Create a function data combines modalities given the type of input.

    Args:
        unimodal (str): Input type as a string. Can be 'force', 'proprio', 'image'. Defaults to using all modalities otherwise
        output (int): Index of output modality.
    """
    def _combine_modalities(data):
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
    return _combine_modalities


def get_data(device, configs, filedirprefix="", unimodal=None, output='contact_next'):
    """Get dataloaders for robotics dataset.

    Args:
        device (torch.utils.device): Device to load data to.
        configs (dict): Configuration dictionary
        filedirprefix (str, optional): File directory prefix path. Defaults to "".
        unimodal (str, optional): Input modality as a string. Defaults to None. Can be 'force', 'proprio', 'image'. Defaults to using all modalities otherwise.
        output (str, optional): Output format. Defaults to 'contact_next'.

    Returns:
        _type_: _description_
    """
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

    image_noise = True
    prop_noise = True
    force_noise = True
    if unimodal == 'image':
        prop_noise = False
        force_noise = False
    elif unimodal == 'proprio':
        image_noise = False
        force_noise = False
    elif unimodal == 'force':
        image_noise = False
        prop_noise = False

    test_image = []
    test_prop = []
    test_force = []
    if image_noise:
        for i in range(10):
            test_image.append(MultimodalManipulationDataset_robust(
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
                noise_level=i/10,
                image_noise=True
            )
            )
    if prop_noise:
        for i in range(10):
            test_prop.append(MultimodalManipulationDataset_robust(
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
                noise_level=i/10,
                prop_noise=True
            )
            )
    if force_noise:
        for i in range(10):
            test_force.append(MultimodalManipulationDataset_robust(
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
                noise_level=i/10,
                force_noise=True
            )
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
    dl_image = []
    for i in range(len(test_image)):
        dl_image.append(
            DataLoader(
                test_image[i],
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                sampler=samplers["val"],
                pin_memory=True,
                drop_last=True,
            )
        )
    dl_prop = []
    for i in range(len(test_prop)):
        dl_prop.append(
            DataLoader(
                test_prop[i],
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                sampler=samplers["val"],
                pin_memory=True,
                drop_last=True,
            )
        )
    dl_force = []
    for i in range(len(test_force)):
        dl_force.append(
            DataLoader(
                test_force[i],
                batch_size=configs['batch_size'],
                num_workers=configs['num_workers'],
                sampler=samplers["val"],
                pin_memory=True,
                drop_last=True,
            )
        )
    dataloaders['test'] = dict()
    if image_noise:
        dataloaders['test']['image'] = dl_image
    if prop_noise:
        dataloaders['test']['proprio'] = dl_prop
    if force_noise:
        dataloaders['test']['force'] = dl_force

    print("Finished setting up date")
    return dataloaders['train'], dataloaders['val'], dataloaders['test']
