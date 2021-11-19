# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py

from training_structures.Supervised_Learning import train, test
from fusions.common_fusions import TensorFusion
from unimodals.gentle_push.head import Head
from unimodals.common_models import Sequential, Transpose, Reshape, Identity, MLP
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
prop_robust_trajectories = Task.get_test_trajectories(
    prop_noise=True, **dataset_args)
haptics_robust_trajectories = Task.get_test_trajectories(
    haptics_noise=True, **dataset_args)
controls_robust_trajectories = Task.get_test_trajectories(
    controls_noise=True, **dataset_args)
multimodal_robust_trajectories = Task.get_test_trajectories(
    multimodal_noise=True, **dataset_args)

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
        SubsequenceDataset(image_robust_trajectories[i], 16),
        batch_size=32,
        shuffle=False,
    ))

prop_robust_loader = []
for i in range(len(prop_robust_trajectories)):
    prop_robust_loader.append(DataLoader(
        SubsequenceDataset(prop_robust_trajectories[i], 16),
        batch_size=32,
        shuffle=False,
    ))

haptics_robust_loader = []
for i in range(len(haptics_robust_trajectories)):
    haptics_robust_loader.append(DataLoader(
        SubsequenceDataset(haptics_robust_trajectories[i], 16),
        batch_size=32,
        shuffle=False,
    ))

controls_robust_loader = []
for i in range(len(controls_robust_trajectories)):
    controls_robust_loader.append(DataLoader(
        SubsequenceDataset(controls_robust_trajectories[i], 16),
        batch_size=32,
        shuffle=False,
    ))

multimodal_robust_loader = []
for i in range(len(multimodal_robust_trajectories)):
    multimodal_robust_loader.append(DataLoader(
        SubsequenceDataset(multimodal_robust_trajectories[i], 16),
        batch_size=32,
        shuffle=False,
    ))

encoders = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(
        8), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape(
        [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(16), Transpose(0, 1)),
]
fusion = TensorFusion()
head = MLP((8 + 1) * (8 + 1) * (64 + 1) * (16 + 1), 256, 2)
allmodules = [*encoders, fusion, head]
optimtype = optim.Adam
loss_state = nn.MSELoss()


def trainprocess(filename):
    train(encoders, fusion, head,
          train_loader, val_loader,
          20,
          task='regression',
          save=filename
          optimtype=optimtype,
          objective=loss_state,
          lr=0.00001)


filename = general_train(trainprocess, 'gentle_push_tensor_fusion')


def testprocess(model, testdata):
    return test(model, testdata, task='regression', criterion=loss_state)


general_test(testprocess, filename, [image_robust_loader, prop_robust_loader,
             haptics_robust_loader, controls_robust_loader, multimodal_robust_loader])
