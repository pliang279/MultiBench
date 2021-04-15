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

class EarlyFusion(nn.Module):
    hidden_size = 128

    def __init__(self, n_features):
        super().__init__()

        self.init_hidden = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.init_cell = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.lstm = nn.LSTM(n_features, self.hidden_size, batch_first=True)
        self.fnn = nn.Linear(self.hidden_size, 1)

    def forward(self, x, training=True):
        if len(x.shape) == 2:
            # unimodal
            x = x.reshape([x.shape[0], x.shape[1], 1])

        _, (h, _) = self.lstm(x,
                              (self.init_hidden.repeat(x.shape[0]).reshape(1, x.shape[0], -1),
                               self.init_cell.repeat(x.shape[0]).reshape(1, x.shape[0], -1)))
        out = self.fnn(h.reshape(-1, self.hidden_size))
        out = out.reshape(out.shape[:1])
        return out

class EarlyFusionFuse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, training=True):
        x = torch.stack(x)
        x = x.permute([1, 2, 0])
        return x

def do_train():
    unimodal_models = [nn.Identity().cuda() for x in stocks]
    multimodal_classification_head = EarlyFusion(len(stocks)).cuda()
    unimodal_classification_heads = [EarlyFusion(1).cuda() for x in stocks]
    fuse = EarlyFusionFuse().cuda()
    training_structures.gradient_blend.criterion = criterion

    train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader=train_loader, valid_dataloader=val_loader,
          classification=False, gb_epoch=2, num_epoch=4, lr=0.001, optimtype=torch.optim.Adam)

#train
do_train()

#test
model = torch.load('best.pt').cuda()
model.eval()
test(model, test_loader, classification=False)
