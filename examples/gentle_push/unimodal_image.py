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
from unimodals.common_models import Sequential, Transpose, Reshape, MLP
from unimodals.gentle_push.head import Head
from fusions.common_fusions import ConcatWithLinear
from training_structures.Supervised_Learning import train, test

Task = PushTask
RobustTask = RobustPushTask
modalities = ['image']

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

train_loader = DataLoader(
    SubsequenceDataset(train_trajectories, 16, modalities),
    batch_size=32,
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    SubsequenceDataset(val_trajectories, 16, modalities),
    batch_size=32,
    shuffle=True,
)
image_robust_loader = []
for i in range(len(image_robust_trajectories)):
    image_robust_loader.append(DataLoader(
            RobustSubsequenceDataset(image_robust_trajectories[i], 16, modalities),
            batch_size=32,
            shuffle=False,
        ))
test_loaders = {
    'image': image_robust_loader,
}

encoders = [
    Sequential(Transpose(0, 1), Reshape([-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64])),
]
fusion = ConcatWithLinear(64, 64, concat_dim=2)
head = Sequential(Head(), Transpose(0, 1))
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
