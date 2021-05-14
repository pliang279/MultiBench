import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from fusions.common_fusions import ConcatWithLinear
from unimodels.common_models import LSTM
from datasets.stocks.get_data import get_dataloader
from training_structures.unimodal import train, test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock])

n_modalities = train_loader.dataset[0][0].size(0)
encoders = [LSTM(1, 16).cuda() for _ in range(n_modalities)]
fusion = ConcatWithLinear(n_modalities * 16).cuda()
head = nn.Identity().cuda()

train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
      task='regression', optimtype=torch.optim.Adam, criterion=nn.MSELoss())

test(torch.load('best.pt').cuda(), test_loader, task='regression')
