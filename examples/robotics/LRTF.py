import sys
import os
sys.path.insert(0, os.getcwd())
from fusions.robotics.sensor_fusion import SensorFusionSelfSupervised, roboticsConcat
from utils.helper_modules import Sequential2
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets.robotics.get_data import get_data
from fusions.common_fusions import LowRankTensorFusion
from robotics_utils import set_seeds
from training_structures.Supervised_Learning import train, test
from unimodals.robotics.decoders import ContactDecoder
from unimodals.common_models import MLP
from unimodals.robotics.encoders import (
    ProprioEncoder, ForceEncoder, ImageEncoder, DepthEncoder, ActionEncoder,
)
from tqdm import tqdm
import yaml
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time



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
        """
        self.fusion = SensorFusionSelfSupervised(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
        ).to(self.device)
        """
        self.fusion = Sequential2(roboticsConcat(
            "noconcat"), LowRankTensorFusion([256, 256, 256, 256, 32], 200, 40))
        #self.head = ContactDecoder(z_dim=configs["zdim"], deterministic=configs["deterministic"])
        self.head = MLP(200, 128, 2)
        self.optimtype = optim.Adam

        # losses
        self.loss_contact_next = nn.BCEWithLogitsLoss()

        self.train_loader, self.val_loader, _ = get_data(
            self.device, self.configs, "/home/pliang/multibench/MultiBench-robotics/")

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
