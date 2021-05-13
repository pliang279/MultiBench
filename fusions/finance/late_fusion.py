import torch
import torch.nn.functional as F
from fusions.common_fusions import ConcatWithLinear
from torch import nn

class LateFusion(nn.Module):
    def __init__(self, n_features, hidden_size=16):
        super().__init__()

        self.hidden_size = hidden_size

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
