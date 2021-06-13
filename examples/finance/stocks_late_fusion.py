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
from unimodals.common_models import LSTM, Identity
from datasets.stocks.get_data import get_dataloader
from training_structures.Supervised_Learning import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock])

n_modalities = len(train_loader.dataset[0]) - 1
encoders = [LSTM(1, 16).cuda() for _ in range(n_modalities)]
fusion = ConcatWithLinear(n_modalities * 16, 1).cuda()
head = Identity().cuda()
allmodules = [*encoders, fusion, head]

def trainprocess():
    train(encoders, fusion, head, train_loader, val_loader, total_epochs=4,
          task='regression', optimtype=torch.optim.Adam, objective=nn.MSELoss())
all_in_one_train(trainprocess, allmodules)

model = torch.load('best.pt').cuda()
# dataset = 'finance F&B', finance tech', finance health'
test(model, test_loader, dataset='finance F&B', task='regression', criterion=nn.MSELoss())
