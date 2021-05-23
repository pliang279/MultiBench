import torch
import torch.nn.functional as F
from fusions.common_fusions import ConcatWithLinear
from torch import nn

class LateFusionTransformer(nn.Module):
    def __init__(self, embed_dim=9):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv = nn.Conv1d(1, self.embed_dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=3)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x, training=False):
        x = self.conv(x.view(x.size(0), 1, -1))
        x = x.permute(2, 0, 1)
        x = self.transformer(x)[-1]
        return x
