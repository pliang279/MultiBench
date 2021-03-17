import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math

class MLPEncoder(torch.nn.Module):
  def __init__(self, indim, hiddim, outdim):
    super(MLPEncoder, self).__init__()
    self.fc = nn.Linear(indim,hiddim)
    self.fc2 = nn.Linear(hiddim,2*outdim)
    self.outdim=outdim
  def forward(self, x, training=False):
    output = self.fc(x)
    output = F.relu(output)
    output = self.fc2(output)
    return output[:,:self.outdim],output[:,self.outdim:]

class TSEncoder(torch.nn.Module):
  def __init__(self,indim,outdim,finaldim,timestep):
    super(TSEncoder,self).__init__()
    self.gru=nn.GRU(input_size=indim,hidden_size=outdim)
    self.indim=indim
    self.ts=timestep
    self.finaldim=finaldim
    self.linear=nn.Linear(outdim*timestep,2*finaldim)
  def forward(self,x,training=False):
    batch=len(x)
    input = x.reshape(batch,self.ts,self.indim).transpose(0,1)
    output= self.gru(input)[0].transpose(0,1)
    output= self.linear(output.flatten(start_dim=1))
    return output[:,:self.finaldim],output[:,self.finaldim:]

class TSDecoder(torch.nn.Module):
  def __init__(self,indim,outdim,finaldim,timestep):
    super(TSDecoder,self).__init__()
    self.gru=nn.GRU(input_size=indim,hidden_size=indim)
    self.linear=nn.Linear(finaldim,indim)
    self.ts=timestep
    self.indim=indim
  def forward(self,x,training=False):
    #print(x.size())
    hidden=self.linear(x).unsqueeze(0)
    next=torch.zeros(1,len(x),self.indim).cuda()
    nexts=[]
    for i in range(self.ts):
      next,hidden=self.gru(next,hidden)
      nexts.append(next.squeeze(0))
    return torch.cat(nexts,1)
