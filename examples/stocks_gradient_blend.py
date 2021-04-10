import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets.stocks.get_data import get_dataloader
from fusions.common_fusions import ConcatWithLinear
from modules.transformer import TransformerEncoder
from training_structures.gradient_blend import train


stocks = sorted(['MCD', 'SBUX', 'HSY', 'HRL'])
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, ['MCD'], modality_first=True, cuda=False)
train_loader = train_loader
val_loader = val_loader
test_loader = test_loader

criterion = nn.MSELoss()

print('Baseline val MSE loss: ' + str(float(nn.MSELoss()(torch.ones_like(val_loader.dataset.Y) * torch.mean(val_loader.dataset.Y), val_loader.dataset.Y))))
print('Baseline test MSE loss: ' + str(float(nn.MSELoss()(torch.ones_like(test_loader.dataset.Y) * torch.mean(test_loader.dataset.Y), test_loader.dataset.Y))))

class EarlyFusion(nn.Module):
    hidden_size = 128

    def __init__(self, n_features):
        super().__init__()

        self.init_hidden = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.init_cell = torch.nn.Parameter(torch.zeros(self.hidden_size))
        self.lstm = nn.LSTM(n_features, self.hidden_size, batch_first=True)
        self.fnn = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x,
                              (self.init_hidden.repeat(x.shape[0]).reshape(1, x.shape[0], -1),
                               self.init_cell.repeat(x.shape[0]).reshape(1, x.shape[0], -1)))
        return self.fnn(h.reshape(-1, self.hidden_size))

class EarlyFusionFuse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.tensor(x).cuda()
        x = x.permute([1, 2, 0])
        return x

def do_train():
    unimodal_models = [nn.Identity() for x in stocks]
    multimodal_classification_head = EarlyFusion(len(stocks))
    unimodal_classification_heads = [EarlyFusion(1) for x in stocks]
    fuse = EarlyFusionFuse()
    train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader=train_loader, valid_dataloader=val_loader,
          num_epoch=80, lr=0.001, optimtype=torch.optim.Adam)

#train
do_train()

#test
model = torch.load('best.pt').cuda()
model.eval()
losses = []
for j, (x, y) in enumerate(test_loader):
    out = model(x)
    loss = criterion(out, y)
    losses.append(loss.item() * len(out))
print(f'Test loss: {np.sum(losses) / len(test_loader.dataset)}')
