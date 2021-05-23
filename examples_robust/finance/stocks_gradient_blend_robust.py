import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import training_structures.gradient_blend
from torch import nn
from fusions.common_fusions import Stack
from unimodals.common_models import LSTMWithLinear
sys.path.append('/home/pliang/multibench/MultiBench/datasets/stocks')
from get_data_robust import get_dataloader
from training_structures.gradient_blend import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test_robust


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

filename = 'stocks_gradient_blend_best.pt'
def trainprocess():
    train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader=train_loader, valid_dataloader=val_loader,
          classification=False, gb_epoch=2, num_epoch=4, lr=0.001, optimtype=torch.optim.Adam, savedir=filename)
all_in_one_train(trainprocess, allmodules)

model = torch.load(filename).cuda()
def testprocess(robust_test_loader):
    return test(model, robust_test_loader, classification=False)
acc = []
print("Robustness testing:")
for noise_level in range(len(test_loader)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(all_in_one_test_robust(testprocess, [model], test_loader[noise_level]))

print("Accuracy of different noise levels:", acc)
