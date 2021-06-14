# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py

import sys
import os
sys.path.insert(0, os.getcwd())

import argparse
import datetime

import fannypack
import torch
import torch.nn as nn
import torch.optim as optim

import unimodals.gentle_push.layers as layers

from torch.utils.data import DataLoader

from datasets.gentle_push.data_loader import SubsequenceDataset, PushTask
from datasets.gentle_push.data_loader_robust import (
    SubsequenceDataset as RobustSubsequenceDataset,
    PushTask as RobustPushTask,
)
from unimodals.common_models import Sequential, Transpose, Reshape, Identity, MLP
from unimodals.gentle_push.head import Head
from fusions.common_fusions import TensorFusion
from training_structures.Supervised_Learning import train, test

Task = PushTask
RobustTask = RobustPushTask

# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)
parser = argparse.ArgumentParser()
RobustTask.add_dataset_arguments(parser)
args = parser.parse_args()
robust_dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('datasets/gentle_push/cache')

# Load trajectories into memory
train_trajectories = Task.get_train_trajectories(**dataset_args)
val_trajectories = Task.get_eval_trajectories(**dataset_args)
image_robust_trajectories = RobustTask.get_test_trajectories(visual_noise=True, **robust_dataset_args)
prop_robust_trajectories = RobustTask.get_test_trajectories(prop_noise=True, **robust_dataset_args)
haptics_robust_trajectories = RobustTask.get_test_trajectories(haptics_noise=True, **robust_dataset_args)
controls_robust_trajectories = RobustTask.get_test_trajectories(controls_noise=True, **robust_dataset_args)
multimodal_robust_trajectories = RobustTask.get_test_trajectories(multimodal_noise=True, **robust_dataset_args)

train_loader = DataLoader(
    SubsequenceDataset(train_trajectories, 16),
    batch_size=32,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    SubsequenceDataset(val_trajectories, 16),
    batch_size=32,
    shuffle=True,
)
image_robust_loader = []
for i in range(len(image_robust_trajectories)):
    image_robust_loader.append(DataLoader(
            RobustSubsequenceDataset(image_robust_trajectories[i], 16),
            batch_size=32,
            shuffle=False,
        ))

prop_robust_loader = []
for i in range(len(prop_robust_trajectories)):
    prop_robust_loader.append(DataLoader(
            RobustSubsequenceDataset(prop_robust_trajectories[i], 16),
            batch_size=32,
            shuffle=False,
        ))

haptics_robust_loader = []
for i in range(len(haptics_robust_trajectories)):
    haptics_robust_loader.append(DataLoader(
            RobustSubsequenceDataset(haptics_robust_trajectories[i], 16),
            batch_size=32,
            shuffle=False,
        ))

controls_robust_loader = []
for i in range(len(controls_robust_trajectories)):
    controls_robust_loader.append(DataLoader(
            RobustSubsequenceDataset(controls_robust_trajectories[i], 16),
            batch_size=32,
            shuffle=False,
        ))

multimodal_robust_loader = []
for i in range(len(multimodal_robust_trajectories)):
    multimodal_robust_loader.append(DataLoader(
            RobustSubsequenceDataset(multimodal_robust_trajectories[i], 16),
            batch_size=32,
            shuffle=False,
        ))
test_loaders = {
    'image': image_robust_loader,
    'prop': prop_robust_loader,
    'haptics': haptics_robust_loader,
    'controls': controls_robust_loader,
    'multimodal': multimodal_robust_loader,
}

encoders = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape([-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(16), Transpose(0, 1)),
]
fusion = TensorFusion()
head = MLP((8 + 1) * (8 + 1) * (64 + 1) * (16 + 1), 256, 2)
optimtype = optim.Adam
loss_state = nn.MSELoss()

train(encoders, fusion, head,
      train_loader, val_loader,
      20,
      task='regression',
      optimtype=optimtype,
      objective=loss_state,
      lr=0.00001)

model = torch.load('best.pt').cuda()
test(model, test_loaders, dataset='gentle push', task='regression', criterion=loss_state)
