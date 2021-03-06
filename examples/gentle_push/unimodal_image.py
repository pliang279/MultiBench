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

from datasets.gentle_push.data_loader import PushTask
from unimodals.common_models import Sequential, Transpose, Reshape, MLP
from unimodals.gentle_push.head import Head
from fusions.common_fusions import ConcatWithLinear
from training_structures.Supervised_Learning import train, test


Task = PushTask
modalities = ['image']

# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('datasets/gentle_push/cache')

train_loader, val_loader, test_loader = Task.get_dataloader(16, modalities, batch_size=32, drop_last=True)

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
test(model, test_loader, dataset='gentle push', task='regression', criterion=loss_state)
