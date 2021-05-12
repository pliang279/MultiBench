import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets.stocks.get_data import get_dataloader
from fusions.finance.early_fusion import EarlyFusionTransformer
from fusions.finance.late_fusion import LateFusionTransformer

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

Model = EarlyFusionTransformer
if args.model == 'late_fusion_transformer':
    Model = LateFusionTransformer

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
