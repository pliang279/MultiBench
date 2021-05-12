import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import pmdarima
import torch
import torch.nn.functional as F
from torch import nn
from fusions.finance.early_fusion import EarlyFusion
from fusions.finance.late_fusion import LateFusion
from fusions.finance.mult import MULTModel
from datasets.stocks.get_data import get_dataloader
from training_structures.unimodal import train, test


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
parser.add_argument('--model', metavar='model', help='model')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)
print('Model: ' + args.model)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, [args.target_stock])

criterion = nn.MSELoss()

def trivial_baselines():
    def best_constant(y_prev, y):
        return float(nn.MSELoss()(torch.ones_like(y) * torch.mean(y), y))
    def copy_last(y_prev, y):
        return nn.MSELoss()(torch.cat([y_prev[-1:], y[:-1]]), y).item()
    def arima(y_prev, y):
        arr = y_prev.cpu()
        arima = pmdarima.arima.auto_arima(arr)
        pred = arima.predict(len(y))
        return nn.MSELoss()(torch.tensor(pred, device='cuda').reshape(y.shape), y)

    print('Best constant val MSE loss: ' + str(best_constant(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('Best constant test MSE loss: ' + str(best_constant(val_loader.dataset.Y, test_loader.dataset.Y)))
    print('Copy-last val MSE loss: ' + str(copy_last(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('Copy-last test MSE loss: ' + str(copy_last(val_loader.dataset.Y, test_loader.dataset.Y)))
    print('ARIMA val MSE loss: ' + str(arima(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('ARIMA test MSE loss: ' + str(arima(torch.cat([train_loader.dataset.Y, val_loader.dataset.Y]), test_loader.dataset.Y)))
trivial_baselines()


Model = EarlyFusion
if args.model == 'late_fusion':
    Model = LateFusion
elif args.model == 'mult':
    Model = MULTModel

train(Model(train_loader.dataset[0][0].shape[1]).cuda(), nn.Identity(),
      train_loader, valid_loader, total_epochs=4, task='regression',
      optimtype=torch.optim.Adam, criterion=nn.MSELoss())

test(torch.load('encoder.pt').cuda(), torch.load('head.pt').cuda(), test_loader, task='regression')
