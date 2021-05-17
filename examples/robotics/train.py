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
from unimodals.robotics.decoders import Decoder
from utils import (
    kl_normal,
    realEPE,
    compute_accuracy,
    flow2rgb,
    set_seeds,
    augment_val,
)

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

        self.encoders = [
            ImageEncoder(configs['zdim'], alpha=configs['vision']),
            DepthEncoder(configs['zdim'], alpha=configs['depth']),
            ProprioEncoder(configs['zdim'], alpha=configs['proprio']),
            ForceEncoder(configs['zdim'], alpha=configs['force']),
            ActionEncoder(configs['action_dim']),
        ]
        self.fusion = SensorFusionSelfSupervised(
            device=self.device,
            encoder=configs["encoder"],
            deterministic=configs["deterministic"],
            z_dim=configs["zdim"],
        ).to(self.device)
        self.head = Decoder(deterministic=configs["deterministic"])

        self.optimtype = optim.Adam

        # losses
        self.loss_ee_pos = nn.MSELoss()
        self.loss_contact_next = nn.BCEWithLogitsLoss()
        self.loss_optical_flow_mask = nn.BCEWithLogitsLoss()
        self.loss_reward_prediction = nn.MSELoss()
        self.loss_is_paired = nn.BCEWithLogitsLoss()
        self.loss_dynamics = nn.MSELoss()

        self.train_loader, self.val_loader = get_data(self.device, self.configs)

    def train(self):
        train(self.encoders, self.fusion, self.head,
              self.train_loader, self.val_loader,
              self.configs['max_epoch'],
              optimtype=self.optimtype)

with open('examples/robotics/training_default.yaml') as f:
    configs = yaml.load(f)

selfsupervised(configs).train()
