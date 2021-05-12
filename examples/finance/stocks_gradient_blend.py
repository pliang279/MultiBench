import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import training_structures.gradient_blend
from torch import nn
from datasets.stocks.get_data import get_dataloader
from fusions.finance.early_fusion import EarlyFusion, EarlyFusionFuse
from training_structures.gradient_blend import train, test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock], modality_first=True, cuda=False)

criterion = nn.MSELoss()

def do_train():
    unimodal_models = [nn.Identity().cuda() for x in stocks]
    multimodal_classification_head = EarlyFusion(len(stocks), single_output=True).cuda()
    unimodal_classification_heads = [EarlyFusion(1, single_output=True).cuda() for x in stocks]
    fuse = EarlyFusionFuse().cuda()
    training_structures.gradient_blend.criterion = criterion

    train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader=train_loader, valid_dataloader=val_loader,
          classification=False, gb_epoch=2, num_epoch=4, lr=0.001, optimtype=torch.optim.Adam)

#train
for i in range(5):
    do_train()

    #test
    model = torch.load('best.pt').cuda()
    model.eval()
    test(model, test_loader, classification=False)
