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

def do_train():
    model = Model(train_loader.dataset[0][0].shape[1]).cuda()
    model.train()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    epochs = 2

    for i in range(epochs):
        losses = []
        for j, (x, y) in enumerate(train_loader):
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if i % 2 == 0:
            print(f'Epoch {i}: {np.mean(losses)}')

    model.eval()
    losses = []
    for j, (x, y) in enumerate(val_loader):
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item() * len(out))

    print(f'Val loss: {np.sum(losses) / len(val_loader.dataset)}')

    return model, np.mean(losses)

#train
best = 9999
for i in range(5):
    model, loss = do_train()
    if loss < best:
        best = loss
        torch.save(model, 'best.pt')

    #test
    model.eval()
    losses = []
    for j, (x, y) in enumerate(test_loader):
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item() * len(out))
    print(f'Test loss: {np.sum(losses) / len(test_loader.dataset)}')
