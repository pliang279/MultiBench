import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from unimodals.common_models import Identity
from fusions.finance.mult import MULTModel
from datasets.stocks.get_data import get_dataloader, Grouping
from training_structures.unimodal import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock], modality_first=False)

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
def testprocess():
    test(encoder, head, test_loader, task='regression', criterion=nn.MSELoss())
all_in_one_test(testprocess, [encoder, head])
