import sys
import os
sys.path.append(os.getcwd())
from datasets.stocks.get_data import get_dataloader
import numpy as np
import torch
from torch import nn

stocks = sorted(['DRI', 'MCD', 'SBUX', 'YUM', 'CPB', 'CAG', 'GIS', 'HSY', 'HRL', 'SJM', 'K', 'MKC', 'TSN'])
train_loader, val_loader, test_loader = get_dataloader(stocks, stocks, ['MCD'])
train_loader = train_loader
val_loader = val_loader
test_loader = test_loader

criterion = nn.MSELoss()

class LSTM(nn.Module):
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

def do_train():
    model = LSTM(n_features=train_loader.dataset[0][0].shape[1]).cuda()
    model.train()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    epochs = 40

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
        out = out * (out > 0)
        loss = criterion(out, y)
        losses.append(loss.item() * len(out))

    print(f'Val loss: {np.sum(losses) / len(val_loader.dataset)}')

    return model, np.mean(losses)

#train
best = 9999
for i in range(10):
    model, loss = do_train()
    if loss < best:
        best = loss
        torch.save(model, 'best.pt')

#test
model = torch.load('best.pt').cuda()
model.eval()
losses = []
for j, (x, y) in enumerate(test_loader):
    out = model(x)
    out = out * (out > 0)
    loss = criterion(out, y)
    losses.append(loss.item() * len(out))
print(f'Test loss: {np.sum(losses) / len(test_loader.dataset)}')
