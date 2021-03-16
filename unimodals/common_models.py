import torch
from torch import nn
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=False,dropoutp=0.1):
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim,hiddim)
        self.fc2 = nn.Linear(hiddim,outdim)
        self.dropoutp = dropoutp
        self.dropout = dropout
    def forward(self, x, training=True):
        output = self.fc(x)
        output = F.dropout(F.relu(output),p=self.dropout,training=training)
        output = self.fc2(output)
        if self.dropout:
            output = F.dropout(output,p=self.dropoutp,training=training)
        return output


class GRU(torch.nn.Module):
    def __init__(self,indim,hiddim,dropout=False,dropoutp=0.1):
        super(GRU,self).__init__()
        self.gru=nn.GRU(indim,hiddim)
        self.dropoutp=dropoutp
        self.dropout=dropout
    def forward(self,x,training=True):
        out=self.gru(x)[0]
        if self.dropout:
            out = F.dropout(out,p=self.dropoutp,training=training)
        return out
