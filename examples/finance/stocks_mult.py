from torch import nn
import torch.nn.functional as F
import torch
import pmdarima
import numpy as np
import argparse
import sys
import os

sys.path.append(os.getcwd())

from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.unimodal import train, test # noqa
from datasets.stocks.get_data import get_dataloader, Grouping # noqa
from fusions.mult import MULTModel # noqa
from unimodals.common_models import Identity # noqa


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(
    stocks, stocks, [args.target_stock], modality_first=False)

n_modalities = 3
grouping = Grouping(n_modalities)

# Get n_features for each group
n_features = [x.size(-1) for x in grouping(next(iter(train_loader))[0])]

model = nn.Sequential(grouping, MULTModel(n_modalities, n_features)).cuda()
identity = Identity()
allmodules = [model, identity]


def trainprocess():
    train(model, identity, train_loader, val_loader,
          total_epochs=4, task='regression',
          optimtype=torch.optim.Adam, criterion=nn.MSELoss())


all_in_one_train(trainprocess, allmodules)

encoder = torch.load('encoder.pt').cuda()
head = torch.load('head.pt').cuda()
# dataset = 'finance F&B', finance tech', finance health'
test(encoder, head, test_loader, dataset='finance F&B',
     task='regression', criterion=nn.MSELoss())
