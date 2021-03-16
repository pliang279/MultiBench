import torch
from torch import nn
from torch.nn import functional as F

class MLP(torch.nn.Module):
  def __init__(self, indim, hiddim, outdim):
    super(MLP, self).__init__()
    self.fc = nn.Linear(indim,hiddim)
    self.fc2 = nn.Linear(hiddim,outdim)
  def forward(self, x, training=True):
    output = self.fc(x)
    output = F.relu(output)
    output = self.fc2(output)
    return output

class MLP_dropout(torch.nn.Module):
  def __init__(self, indim, hiddim, outdim, dropout=0.1):
    super(MLP_dropout, self).__init__()
    self.fc = nn.Linear(indim,hiddim)
    self.fc2 = nn.Linear(hiddim,outdim)
    self.dropout=dropout
  def forward(self, x, training=True):
    output = self.fc(x)
    output = F.dropout(F.relu(output),p=self.dropout,training=training)
    output = self.fc2(output)
    return F.dropout(output,p=self.dropout,training=training)

class GRU(torch.nn.Module):
    def __init__(self,indim,hiddim):
        super(GRU,self).__init__()
        self.gru=nn.GRU(indim,hiddim)
    def forward(self,x,training=True):
        return self.gru(x)[0]

class GRU_dropout(torch.nn.Module):
    def __init__(self,indim,hiddim,dropout=0.1):
        super(GRU_dropout,self).__init__()
        self.gru=nn.GRU(indim,hiddim)
        self.dropout=dropout
    def forward(self,x,training=True):
        return F.dropout(self.gru(x)[0],p=self.dropout,training=training)
