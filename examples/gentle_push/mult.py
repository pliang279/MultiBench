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
from unimodals.common_models import Sequential, Transpose, Reshape, Identity
from unimodals.gentle_push.head import Head
from fusions.mult import MULTModel
from training_structures.Supervised_Learning import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from xy_mse_loss import XYMSELoss

Task = PushTask

# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('datasets/gentle_push/cache')

# Load trajectories into memory
train_trajectories = Task.get_train_trajectories(**dataset_args)
val_trajectories = Task.get_eval_trajectories(**dataset_args)
test_trajectories = Task.get_test_trajectories(**dataset_args)

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
test_loader = DataLoader(
    SubsequenceDataset(test_trajectories, 16),
    batch_size=32,
    shuffle=False,
)

class HyperParams(MULTModel.DefaultHyperParams):
    num_heads = 4
    embed_dim = 64
    output_dim = 2
    all_steps = True

encoders = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(64), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(64), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape([-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(64), Transpose(0, 1)),
]
fusion = MULTModel(4, [64, 64, 64, 64], HyperParams)
head = Identity()
allmodules = [*encoders, fusion, head]
optimtype = optim.Adam
loss_state = nn.MSELoss()

def trainprocess():
    train(encoders, fusion, head,
          train_loader, val_loader,
          20,
          task='regression',
          optimtype=optimtype,
          objective=loss_state,
          lr=0.00001)
all_in_one_train(trainprocess, allmodules)

model = torch.load('best.pt').cuda()
test(model, test_loader, dataset='gentle push', task='regression', criterion=XYMSELoss())
