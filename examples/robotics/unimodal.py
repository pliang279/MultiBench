import sys
import os
sys.path.append(os.getcwd())

import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from tqdm import tqdm

from fusions.robotics.sensor_fusion import SensorFusionSelfSupervised
from unimodals.robotics.encoders import (
    ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder,
)
from unimodals.robotics.decoders import ContactDecoder
from training_structures.unimodal import train, test
from robotics_utils import set_seeds

from datasets.robotics.data_loader import get_data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms


class selfsupervised:
    def __init__(self, configs):

        # ------------------------
        # Sets seed and cuda
        # ------------------------
        use_cuda = True

        self.configs = configs
        self.device = torch.device("cuda" if use_cuda else "cpu")

        set_seeds(configs["seed"], use_cuda)

        uniencoder = 'image'
        if 'unimodal_encoder' in configs:
            uniencoder = configs['unimodal_encoder']
        if uniencoder == 'image':
            self.encoder = ImageEncoder(configs['zdim'], alpha=configs['vision'])
        elif uniencoder == 'force':
            self.encoder = ForceEncoder(configs['zdim'], alpha=configs['force'])
        elif uniencoder == 'proprio':
            self.encoder = ProprioEncoder(configs['zdim'], alpha=configs['proprio'])
        elif uniencoder == 'depth':
            self.encoder = DepthEncoder(configs['zdim'], alpha=configs['depth'])
        elif uniencoder == 'action':
            self.encoder = ActionEncoder(configs['action_dim'])
        else:
            raise Exception(f'encoder {uniencoder} not found')
        self.head = ContactDecoder(z_dim=2 * configs["zdim"], deterministic=configs["deterministic"])

        self.optimtype = optim.Adam

        # losses
        self.loss_contact_next = nn.BCEWithLogitsLoss()

        self.train_loader, self.val_loader = get_data(self.device, self.configs)

    def train(self):
        train(self.encoder, self.head,
              self.train_loader, self.val_loader,
              self.configs['max_epoch'],
              optimtype=self.optimtype,
              lr=self.configs['lr'],
              criterion=self.loss_contact_next)

with open('examples/robotics/training_default.yaml') as f:
    configs = yaml.load(f)

selfsupervised(configs).train()
