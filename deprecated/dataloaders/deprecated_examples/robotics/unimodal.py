from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets.robotics.data_loader import get_data
from robotics_utils import set_seeds
from training_structures.unimodal import train, test
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

        uniencoder = 'image'
        if 'unimodal_encoder' in configs:
            uniencoder = configs['unimodal_encoder']
        if uniencoder == 'image':
            self.encoder = ImageEncoder(
                configs['zdim'], alpha=configs['vision'])
        elif uniencoder == 'force':
            self.encoder = ForceEncoder(
                configs['zdim'], alpha=configs['force'])
        elif uniencoder == 'proprio':
            self.encoder = ProprioEncoder(
                configs['zdim'], alpha=configs['proprio'])
        elif uniencoder == 'depth':
            self.encoder = DepthEncoder(
                configs['zdim'], alpha=configs['depth'])
        elif uniencoder == 'action':
            self.encoder = ActionEncoder(configs['action_dim'])
        else:
            raise Exception(f'encoder {uniencoder} not found')
        self.head = ContactDecoder(
            z_dim=2 * configs["zdim"], deterministic=configs["deterministic"])

        self.allmodules = [self.encoder, self.head]

        self.optimtype = optim.Adam
        self.loss_contact_next = nn.BCEWithLogitsLoss()
        self.train_loader, self.val_loader = get_data(
            self.device, self.configs)

    def train(self):
        def trainprocess():
            train(self.encoder, self.head,
                  self.train_loader, self.val_loader,
                  self.configs['max_epoch'],
                  optimtype=self.optimtype,
                  lr=self.configs['lr'],
                  criterion=self.loss_contact_next)
        all_in_one_train(trainprocess, self.allmodules)


with open('examples/robotics/training_default.yaml') as f:
    configs = yaml.load(f)

selfsupervised(configs).train()
