import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from fusions.common_fusions import Stack
from unimodals.common_models import LSTMWithLinear
from datasets.stocks.get_data import get_dataloader
from training_structures.Simple_Late_Fusion import train, test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock])

n_modalities = train_loader.dataset[0][0].size(0)
encoders = [nn.Identity().cuda()] * n_modalities
fusion = Stack().cuda()
head = LSTMWithLinear(n_modalities, 128, 1).cuda()

train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
      task='regression', optimtype=torch.optim.Adam, criterion=nn.MSELoss())

test(torch.load('best.pt').cuda(), test_loader, task='regression')
