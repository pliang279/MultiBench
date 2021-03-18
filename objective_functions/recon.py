import torch
from torch import nn

sigm=nn.Sigmoid()

# this is used for MIMIC MVAE
def sigmloss1d(a,b):
  #print(a.size())
  #print(b.size())
  x=sigm(a)
  y=sigm(b)
  ret=torch.mean(-y*torch.log(x)-(1-y)*torch.log(1-x),dim=1)
  #ret=torch.mean(torch.clamp(x,0)-x*y+torch.log(1+torch.exp(-torch.abs(x))),dim=1)
  #if math.isnan(torch.sum(ret).item()) and not printed:
    #print("bce")
    #printed=True
  return ret



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
        totalloss += weights[i]*modal_loss_funcs[i](recons[i],origs[i])
    return torch.mean(totalloss+annealing*kld)
  return actualfunc

def recon_weighted_sum(modal_loss_funcs,weights):
    def actualfunc(recons,origs):
        totalloss=0.0
        for i in range(len(recons)):
            totalloss += modal_loss_funcs[i](recons[i],origs[i])*weights[i]
        return torch.mean(totalloss)
    return actualfunc
