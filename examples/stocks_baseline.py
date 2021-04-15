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

def trivial_baselines():
    def best_constant(y_prev, y):
        return float(nn.MSELoss()(torch.ones_like(y) * torch.mean(y), y))
    def copy_last(y_prev, y):
        return nn.MSELoss()(torch.cat([y_prev[-1:], y[:-1]]), y).item()
    print('Best constant val MSE loss: ' + str(best_constant(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('Best constant test MSE loss: ' + str(best_constant(val_loader.dataset.Y, test_loader.dataset.Y)))
    print('Copy-last val MSE loss: ' + str(copy_last(train_loader.dataset.Y, val_loader.dataset.Y)))
    print('Copy-last test MSE loss: ' + str(copy_last(val_loader.dataset.Y, test_loader.dataset.Y)))
trivial_baselines()

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

class LateFusion(nn.Module):
    hidden_size = 16

    def __init__(self, n_features):
        super().__init__()

        self.init_hidden = torch.nn.Parameter(torch.zeros(n_features, self.hidden_size))
        self.init_cell = torch.nn.Parameter(torch.zeros(n_features, self.hidden_size))
        lstms = []
        for i in range(n_features):
            lstms.append(nn.LSTM(1, self.hidden_size, batch_first=True))
        self.lstms = nn.ModuleList(lstms)
        self.fusion = ConcatWithLinear(n_features * self.hidden_size, 1)

    def forward(self, x):
        hidden = []
        for i in range(len(self.lstms)):
            _, (h, _) = self.lstms[i](x[:, :, i:i + 1],
                                      (self.init_hidden[i].repeat(x.shape[0]).reshape(1, x.shape[0], -1),
                                       self.init_cell[i].repeat(x.shape[0]).reshape(1, x.shape[0], -1)))
            hidden.append(h.reshape(x.shape[0], self.hidden_size))
        return self.fusion(hidden)

class MULTModel(nn.Module):
    # https://github.com/yaohungt/Multimodal-Transformer

    class default_hyp_params():
        num_heads = 3
        layers = 3
        attn_dropout = 0.1
        attn_dropout_modalities = [0.0] * 1000
        relu_dropout = 0.1
        res_dropout = 0.1
        out_dropout = 0.0
        embed_dropout = 0.25
        attn_mask = True
        output_dim = 1

    def __init__(self, n_modalities, hyp_params=default_hyp_params):
        """
        Construct a MulT model.
        """
        super().__init__()
        self.embed_dim = 9
        self.n_modalities = n_modalities
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_modalities = hyp_params.attn_dropout_modalities
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.embed_dim * n_modalities * n_modalities

        output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj = [nn.Conv1d(1, self.embed_dim, kernel_size=1, padding=0, bias=False) for i in range(n_modalities)]
        self.proj = nn.ModuleList(self.proj)

        # 2. Crossmodal Attentions
        self.trans = [nn.ModuleList([self.get_network(i, j, mem=False) for j in range(n_modalities)]) for i in range(n_modalities)]
        self.trans = nn.ModuleList(self.trans)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        self.trans_mems = [self.get_network(i, i, mem=True, layers=3) for i in range(n_modalities)]
        self.trans_mems = nn.ModuleList(self.trans_mems)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, mod1, mod2, mem, layers=-1):
        if not mem:
            embed_dim = self.embed_dim
            attn_dropout = self.attn_dropout_modalities[mod2]
        else:
            embed_dim = self.n_modalities * self.embed_dim
            attn_dropout = self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x):
        """
        x: [batch_size, seq_len, n_modalities]
        """
        x = x.permute(2, 0, 1)
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))

        # Project the textual/visual/audio features
        proj_x = [self.proj[i](x[i]) for i in range(self.n_modalities)]
        proj_x = torch.stack(proj_x)
        proj_x = proj_x.permute(0, 3, 1, 2)

        last_hs = []
        for i in range(self.n_modalities):
            h = []
            for j in range(self.n_modalities):
                h.append(self.trans[i][j](proj_x[i], proj_x[j], proj_x[j]))
            h = torch.cat(h, dim=2)
            h = self.trans_mems[i](h)
            if type(h) == tuple:
                h = h[0]
            last_hs.append(h[-1])

        last_hs = torch.cat(last_hs, dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output

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
model = torch.load('best.pt').cuda()
model.eval()
losses = []
for j, (x, y) in enumerate(test_loader):
    out = model(x)
    loss = criterion(out, y)
    losses.append(loss.item() * len(out))
print(f'Test loss: {np.sum(losses) / len(test_loader.dataset)}')
