import torch
from torch import nn
import math
sigm=nn.Sigmoid()

# this is used for MIMIC MVAE
def sigmloss1d(a,b):
  #print(a)
  #print(b.size())
  #print(a)
  x=sigm(a)
  y=sigm(b)
  #print(x)
  #print(b)
  ret=torch.mean(-y*torch.log(x)-(1-y)*torch.log(1-x),dim=1)
  #print(ret)
  #ret=torch.mean(torch.clamp(x,0)-x*y+torch.log(1+torch.exp(-torch.abs(x))),dim=1)
  #if math.isnan(torch.sum(ret).item()) and not printed:
    #print("bce")
    #printed=True
  return ret

def nosigmloss1d(a,b):
  #print(a.size())
  #print(b.size())
  x=a
  y=b
  ret=torch.mean(-y*torch.log(x)-(1-y)*torch.log(1-x),dim=1)
  #ret=torch.mean(torch.clamp(x,0)-x*y+torch.log(1+torch.exp(-torch.abs(x))),dim=1)
  #if math.isnan(torch.sum(ret).item()) and not printed:
    #print("bce")
    #printed=True
  return ret

def sigmloss1dcentercrop(adim,bdim):
    borderdim = (bdim-adim)//2
    def func(a,b):
        if a.size()[2]>b.size()[2]:
            a1=b
            b1=a
        else:
            a1=a
            b1=b
        #print(a1)
        br = b1[:,:,borderdim:bdim-borderdim,borderdim:bdim-borderdim]
        af = torch.flatten(a1,start_dim=1)
        bf = torch.flatten(br,start_dim=1)
        return sigmloss1d(af,bf)
    return func



def elbo_loss(modal_loss_funcs,weights,annealing=1.0):
  def actualfunc(recons,origs,mu,logvar):
    totalloss=0.0
    if torch.max(logvar).item()>99999:
      kld=logvar
    else:
      kld=-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
      #print(origs)
    for i in range(len(recons)):
      if recons[i] is not None:
        #print(origs[i])
        #print(recons[i])
        totalloss += weights[i]*modal_loss_funcs[i](recons[i],origs[i])
        #print(totalloss)
        #if math.isnan(torch.sum(totalloss).item()):
            #exit(0)
    return torch.mean(totalloss+annealing*kld)
  return actualfunc

def recon_weighted_sum(modal_loss_funcs,weights):
    def actualfunc(recons,origs):
        totalloss=0.0
        for i in range(len(recons)):
            totalloss += modal_loss_funcs[i](recons[i],origs[i])*weights[i]
        return torch.mean(totalloss)
    return actualfunc
