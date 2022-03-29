from torch import nn
import training_structures.gradient_blend
import torch.nn.functional as F
import torch
import numpy as np
import argparse
import sys
import os

sys.path.append(os.getcwd())

from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from training_structures.gradient_blend import train, test # noqa
from datasets.stocks.get_data import get_dataloader # noqa
from unimodals.common_models import LSTM, Identity, Squeeze # noqa
from fusions.common_fusions import Stack # noqa


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


class IgnoreTrainingArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if 'training' in kwargs:
            del kwargs['training']
        return self.module(*args, **kwargs)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(
    stocks, stocks, [args.target_stock], cuda=False)

unimodal_models = [Identity().cuda() for x in stocks]
multimodal_head = IgnoreTrainingArg(nn.Sequential(
    LSTM(len(stocks), 128, linear_layer_outdim=1), Squeeze())).cuda()
unimodal_heads = [IgnoreTrainingArg(nn.Sequential(
    LSTM(1, 128, linear_layer_outdim=1), Squeeze())).cuda() for x in stocks]
fuse = Stack().cuda()
allmodules = [*unimodal_models, multimodal_head, *unimodal_heads, fuse]

training_structures.gradient_blend.criterion = nn.MSELoss()


def trainprocess():
    train(unimodal_models,  multimodal_head,
          unimodal_heads, fuse, train_dataloader=train_loader, valid_dataloader=val_loader,
          classification=False, gb_epoch=2, num_epoch=4, lr=0.001, optimtype=torch.optim.Adam)


all_in_one_train(trainprocess, allmodules)

model = torch.load('best.pt').cuda()
# dataset = 'finance F&B', finance tech', finance health'
test(model, test_loader, dataset='finance F&B', classification=False)
