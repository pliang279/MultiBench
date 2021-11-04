# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py

from training_structures.Supervised_Learning import train, test
from fusions.common_fusions import ConcatWithLinear
from unimodals.gentle_push.head import Head
from unimodals.common_models import Sequential, Transpose, Reshape, MLP
from robustness.all_in_one import general_train, general_test
from datasets.gentle_push.data_loader_robust import SubsequenceDataset, PushTask
from torch.utils.data import DataLoader
import unimodals.gentle_push.layers as layers
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack
import datetime
import argparse
import sys
import os
sys.path.insert(0, os.getcwd())


Task = PushTask
modalities = ['image']

# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('datasets/gentle_push/cache')

# Load trajectories into memory
train_trajectories = Task.get_train_trajectories(**dataset_args)
val_trajectories = Task.get_eval_trajectories(**dataset_args)
image_robust_trajectories = Task.get_test_trajectories(
    visual_noise=True, **dataset_args)

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
        SubsequenceDataset(image_robust_trajectories[i], 16, modalities),
        batch_size=32,
        shuffle=False,
    ))

encoders = [
    Sequential(Transpose(0, 1), Reshape(
        [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64])),
]
fusion = ConcatWithLinear(64, 64, concat_dim=2)
head = Sequential(Head(), Transpose(0, 1))
allmodules = [*encoders, fusion, head]
optimtype = optim.Adam
loss_state = nn.MSELoss()


def trainprocess(filename):
    train(encoders, fusion, head,
          train_loader, val_loader,
          20,
          task='regression',
          save=filename,
          optimtype=optimtype,
          objective=loss_state,
          lr=0.00001)


filename = general_train(trainprocess, 'gentle_push_unimodal_image')


def testprocess(model, testdata):
    return test(model, testdata, task='regression', criterion=loss_state)


general_test(testprocess, filename, [image_robust_loader])
