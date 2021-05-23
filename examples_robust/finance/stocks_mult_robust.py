import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from unimodals.common_models import Identity
from fusions.finance.mult import MULTModel
sys.path.append('/home/pliang/multibench/MultiBench/datasets/stocks')
from get_data_robust import get_dataloader
from training_structures.unimodal import train, test
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test_robust


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock], modality_first=False)

model = MULTModel(train_loader.dataset[0][0].shape[1]).cuda()
identity = Identity()
allmodules = [model, identity]

filename_encoder = 'stocks_mult_encoder.pt'
filename_head = 'stocks_mult_head.pt'
def trainprocess():
    train(model, identity,
          train_loader, val_loader, total_epochs=4, task='regression',
          optimtype=torch.optim.Adam, criterion=nn.MSELoss(), save_encoder=filename_encoder,save_head=filename_head)
all_in_one_train(trainprocess, allmodules)

encoder = torch.load(filename_encoder).cuda()
head = torch.load(filename_head).cuda()
def testprocess(robust_test_loader):
    return test(encoder, head, robust_test_loader, task='regression', criterion=nn.MSELoss())
acc = []
print("Robustness testing:")
for noise_level in range(len(test_loader)):
    print("Noise level {}: ".format(noise_level/10))
    acc.append(test(encoder, head, test_loader[noise_level]))

print("Accuracy of different noise levels:", acc)
