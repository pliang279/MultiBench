import torch
from torch import nn

class Sequential2(nn.Module):
    def __init__(self,a,b):
        super(Sequential2,self).__init__()
        self.model = nn.Sequential(a,b)
    def forward(self,x,training=False):
        return self.model(x)
