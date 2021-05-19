import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from fusions.finance.mult import MULTModel
from datasets.stocks.get_data import get_dataloader
from training_structures.unimodal import train, test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock], modality_first=False)

train(MULTModel(train_loader.dataset[0][0].shape[1]).cuda(), nn.Identity(),
      train_loader, val_loader, total_epochs=4, task='regression',
      optimtype=torch.optim.Adam, criterion=nn.MSELoss())

test(torch.load('encoder.pt').cuda(), torch.load('head.pt').cuda(), test_loader, task='regression')
