import torch
import torch.nn.functional as F
from torch import nn

class EarlyFusion(nn.Module):
    def __init__(self, n_features, hidden_size=128, single_output=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.single_output = single_output

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
        if self.single_output:
            out = out.reshape(out.shape[:-1])
        return out

class EarlyFusionFuse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, training=True):
        x = torch.stack(x)
        x = x.permute([1, 2, 0])
        return x

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
