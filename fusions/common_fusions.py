import torch
from torch import nn
from torch.nn import functional as F

class Concat(nn.Module):
    def __init__(self):
        super(Concat,self).__init__()
    def forward(self,modals,training=False):
        flattened=[]
        for modal in modals:
            flattened.append(torch.flatten(modal,start_dim=1))
        return torch.cat(flattened,dim=1)


class ConcatWithLinear(nn.Module):
    def __init__(self,indim,outdim):
        super(Concat,self).__init__()
        self.fc=nn.Linear(indim,outdim)
    def forward(self,modals,training=False):
        return self.fc(torch.cat(modals,dim=1))

class NLgate(torch.nn.Module):
  # q_linear,k_liear,v_linear are none of no linear layer applied before q,k,v; otherwise, a tuple of (indim,outdim)
  # is inputted for each of these 3 arguments
  # See section F4 of "What makes training MM classification networks hard for details"
  def __init__(self,thw_dim,c_dim,tf_dim, q_linear=None, k_linear=None, v_linear=None):
    super(NLgate,self).__init__()
    self.qli = None
    if q_linear is not None:
      self.qli=nn.Linear(q_linear[0],q_linear[1])
    self.kli = None
    if k_linear is not None:
      self.kli=nn.Linear(k_linear[0],k_linear[1])
    self.vli = None
    if v_linear is not None:
      self.vli=nn.Linear(v_linear[0],v_linear[1])
    self.thw_dim = thw_dim
    self.c_dim = c_dim
    self.tf_dim = tf_dim
    self.softmax = nn.Softmax(dim = 2)
  def forward(self,q,k,v,training=False):
    if self.qli is None:
      qin = q.view(-1,self.thw_dim,self.c_dim)
    else:
      qin = self.qli(q).view(-1,self.thw_dim,self.c_dim)
    if self.kli is None:
      kin = k.view(-1,c_dim,tf_dim)
    else:
      kin = self.kli(k).view(-1,self.c_dim,self.tf_dim)
    if self.vli is None:
      vin = v.view(-1,tf_dim,c_dim)
    else:
      vin = self.vli(v).view(-1,self.tf_dim,self.c_dim)
    matmulled = torch.matmul(qin,kin)
    finalout = torch.matmul(self.softmax(matmulled),vin)
    return qin + finalout



