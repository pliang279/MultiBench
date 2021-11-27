# From https://github.com/brentyi/multimodalfilter/blob/master/scripts/push_task/train_push.py
import torch.optim as optim
import torch.nn as nn
import torch
import fannypack
import datetime
import argparse
import sys
import os

sys.path.insert(0, os.getcwd())

from training_structures.Supervised_Learning import train, test # noqa
from fusions.mult import MULTModel # noqa
from unimodals.gentle_push.head import Head # noqa
from unimodals.common_models import Sequential, Transpose, Reshape, Identity # noqa
from datasets.gentle_push.data_loader import PushTask # noqa
import unimodals.gentle_push.layers as layers # noqa



Task = PushTask

# Parse args
parser = argparse.ArgumentParser()
Task.add_dataset_arguments(parser)
args = parser.parse_args()
dataset_args = Task.get_dataset_args(args)

fannypack.data.set_cache_path('datasets/gentle_push/cache')

train_loader, val_loader, test_loader = Task.get_dataloader(
    16, batch_size=32, drop_last=True)


class HyperParams(MULTModel.DefaultHyperParams):
    num_heads = 4
    embed_dim = 64
    output_dim = 2
    all_steps = True


encoders = [
    Sequential(Transpose(0, 1), layers.observation_pos_layers(
        64), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.observation_sensors_layers(
        64), Transpose(0, 1)),
    Sequential(Transpose(0, 1), Reshape(
        [-1, 1, 32, 32]), layers.observation_image_layers(64), Reshape([16, -1, 64]), Transpose(0, 1)),
    Sequential(Transpose(0, 1), layers.control_layers(64), Transpose(0, 1)),
]
fusion = MULTModel(4, [64, 64, 64, 64], HyperParams)
head = Identity()
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
test(model, test_loader, dataset='gentle push',
     task='regression', criterion=loss_state)
