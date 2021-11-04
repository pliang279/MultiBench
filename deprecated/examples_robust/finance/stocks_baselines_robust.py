from get_data_robust import get_dataloader
from unimodals.common_models import LSTM
from fusions.common_fusions import Stack
from torch import nn
import torch.nn.functional as F
import torch
import pmdarima
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

sys.path.append('/home/pliang/multibench/MultiBench/datasets/stocks')


parser = argparse.ArgumentParser()
parser.add_argument('--input-stocks', metavar='input', help='input stocks')
parser.add_argument('--target-stock', metavar='target', help='target stock')
args = parser.parse_args()
print('Input: ' + args.input_stocks)
print('Target: ' + args.target_stock)


stocks = sorted(args.input_stocks.split(' '))
train_loader, val_loader, test_loader = get_dataloader(
    stocks, stocks, [args.target_stock])


def baselines(robust_test_loader):
    def best_constant(y_prev, y):
        return float(nn.MSELoss()(torch.ones_like(y) * torch.mean(y), y))

    def copy_last(y_prev, y):
        return nn.MSELoss()(torch.cat([y_prev[-1:], y[:-1]]), y).item()

    def arima(y_prev, y):
        arr = y_prev.cpu()
        arima = pmdarima.arima.auto_arima(arr)
        pred = arima.predict(len(y))
        return nn.MSELoss()(torch.tensor(pred, device='cuda').reshape(y.shape), y).item()

    print('Best constant val MSE loss: ' +
          str(best_constant(train_loader.dataset.Y, val_loader.dataset.Y)))
    best_constant_test = best_constant(
        val_loader.dataset.Y, robust_test_loader.dataset.Y)
    print('Best constant test MSE loss: ' + str(best_constant_test))
    print('Copy-last val MSE loss: ' +
          str(copy_last(train_loader.dataset.Y, val_loader.dataset.Y)))
    copy_last_test = copy_last(
        val_loader.dataset.Y, robust_test_loader.dataset.Y)
    print('Copy-last test MSE loss: ' + str(copy_last_test))
    print('ARIMA val MSE loss: ' +
          str(arima(train_loader.dataset.Y, val_loader.dataset.Y)))
    arima_test = arima(torch.cat(
        [train_loader.dataset.Y, val_loader.dataset.Y]), robust_test_loader.dataset.Y)
    print('ARIMA test MSE loss: ' + str(arima_test))

    return best_constant_test, copy_last_test, arima_test


best_constant_loss = []
copy_last_loss = []
arima_loss = []
print("Robustness testing:")
for noise_level in range(len(test_loader)):
    print("Noise level {}: ".format(noise_level/10))
    best_constant, copy_last, arima = baselines(test_loader[noise_level])
    best_constant_loss.append(best_constant)
    copy_last_loss.append(copy_last)
    arima_loss.append(arima)

print("Accuracy of different noise levels (Best constant):", best_constant_loss)
print("Accuracy of different noise levels (Copy-last):", copy_last_loss)
print("Accuracy of different noise levels (Arima):", arima_loss)
