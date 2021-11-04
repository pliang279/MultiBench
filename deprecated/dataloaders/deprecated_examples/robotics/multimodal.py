from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
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
            z_dim=configs["zdim"], deterministic=configs["deterministic"])

        self.allmodules = [*self.encoders, self.fusion, self.head]

        self.optimtype = optim.Adam
        self.loss_contact_next = nn.BCEWithLogitsLoss()
        self.train_loader, self.val_loader = get_data(
            self.device, self.configs)

    def train(self):
        def trainprocess():
            train(self.encoders, self.fusion, self.head,
                  self.train_loader, self.val_loader,
                  self.configs['max_epoch'],
                  optimtype=self.optimtype,
                  lr=self.configs['lr'],
                  criterion=self.loss_contact_next)
        all_in_one_train(trainprocess, self.allmodules)


with open('examples/robotics/training_default.yaml') as f:
    configs = yaml.load(f)

selfsupervised(configs).train()
