import sys
import os
sys.path.append(os.getcwd())

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets.stocks.get_data import get_dataloader
from fusions.common_fusions import ConcatWithLinear
from modules.transformer import TransformerEncoder


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

class EarlyFusionTransformer(nn.Module):
    embed_dim = 9

    def __init__(self, n_features):
        super().__init__()

        self.conv = nn.Conv1d(n_features, self.embed_dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.linear = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x = self.transformer(x)[-1]
        return self.linear(x)

class LateFusionTransformer(nn.Module):
    embed_dim = 9

    def __init__(self, n_features):
        super().__init__()

        convs = [nn.Conv1d(1, self.embed_dim, kernel_size=1, padding=0, bias=False) for _ in range(n_features)]
        self.convs = nn.ModuleList(convs)

        transformers = []
        for i in range(n_features):
            layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
            transformer = nn.TransformerEncoder(layer, num_layers=3)
            transformers.append(transformer)
        self.transformers = nn.ModuleList(transformers)
        self.fusion = ConcatWithLinear(n_features * self.embed_dim, 1)

    def forward(self, x):
        out = []
        for i in range(len(self.transformers)):
            emb = x[:, :, i:i + 1]
            emb = self.convs[i](emb.permute([0, 2, 1]))
            emb = emb.permute([2, 0, 1])
            emb = self.transformers[i](emb)
            emb = emb[-1]
            out.append(emb)
        return self.fusion(out)


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
