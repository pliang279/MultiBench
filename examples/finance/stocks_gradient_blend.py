import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import training_structures.gradient_blend
from torch import nn
from fusions.common_fusions import Stack
from unimodals.common_models import LSTMWithLinear
from datasets.stocks.get_data import get_dataloader
from training_structures.gradient_blend import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock], cuda=False)

unimodal_models = [nn.Identity().cuda() for x in stocks]
multimodal_classification_head = LSTMWithLinear(len(stocks), 128, 1).cuda()
unimodal_classification_heads = [LSTMWithLinear(1, 128, 1).cuda() for x in stocks]
fuse = Stack().cuda()
allmodules = [*unimodal_models, multimodal_classification_head, *unimodal_classification_heads, fuse]

training_structures.gradient_blend.criterion = nn.MSELoss()

def trainprocess():
    train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader=train_loader, valid_dataloader=val_loader,
          classification=False, gb_epoch=2, num_epoch=4, lr=0.001, optimtype=torch.optim.Adam)
all_in_one_train(trainprocess, allmodules)

model = torch.load('best.pt').cuda()
def testprocess():
    test(model, test_loader, classification=False)
all_in_one_test(testprocess, [model])
