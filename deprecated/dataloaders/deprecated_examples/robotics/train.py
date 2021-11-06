from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets.robotics.data_loader import get_data
from robotics_utils import set_seeds
from training_structures.Simple_Late_Fusion import train, test
from unimodals.robotics.decoders import ContactDecoder
from unimodals.robotics.encoders import (
    ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder,
)
from fusions.robotics.sensor_fusion import SensorFusionSelfSupervised
from tqdm import tqdm
import yaml
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time
import sys
import os
sys.path.insert(0, os.getcwd())


class selfsupervised:
    def __init__(self, configs):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = True

        self.configs = configs
        self.device = torch.device("cuda" if use_cuda else "cpu")

        set_seeds(configs["seed"], use_cuda)

        self.encoders = [
            ImageEncoder(configs['zdim'], alpha=configs['vision']),
            ForceEncoder(configs['zdim'], alpha=configs['force']),
            ProprioEncoder(configs['zdim'], alpha=configs['proprio']),
            DepthEncoder(configs['zdim'], alpha=configs['depth']),
            ActionEncoder(configs['action_dim']),
        ]
        self.fusion = SensorFusionSelfSupervised(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
        ).to(self.device)
        self.head = ContactDecoder(
            z_dim=configs["zdim"], deterministic=configs["deterministic"], head=2)

        self.optimtype = optim.Adam

        # losses
        self.loss_contact_next = nn.BCEWithLogitsLoss()

        self.train_loader, self.val_loader = get_data(
            self.device, self.configs)
        for j in self.train_loader:
            print(j[0].size())
            print(j[1].size())
            print(j[2].size())
            print(j[3].size())
            print(j[4].size())

            print(j[5].size())

    def train(self):
        print(len(self.train_loader.dataset), len(self.val_loader.dataset))
        with open('train_dataset.txt', 'w') as f:
            for x in self.train_loader.dataset.dataset_path:
                f.write(f'{x}\n')
        with open('val_dataset.txt', 'w') as f:
            for x in self.val_loader.dataset.dataset_path:
                f.write(f'{x}\n')
        train(self.encoders, self.fusion, self.head,
              self.train_loader, self.val_loader,
              15,
              optimtype=self.optimtype,
              lr=self.configs['lr'])


with open('examples/robotics/training_default.yaml') as f:
    configs = yaml.load(f)

selfsupervised(configs).train()
