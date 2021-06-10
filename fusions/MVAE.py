import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """
    def __init__(self,size):
        super(ProductOfExperts,self).__init__()
        self.size=size
    def forward(self, mus, logvars, eps=1e-8, training=False):
        mu,logvar = prior_expert(self.size,len(mus[0]))
        for i in range(len(mus)):
            #print(mu.shape)
            mu=torch.cat((mu,mus[i].unsqueeze(0)),dim=0)
            logvar=torch.cat((logvar,logvars[i].unsqueeze(0)),dim=0)

        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

class ProductOfExperts_Zipped(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """
    def __init__(self,size):
        super(ProductOfExperts_Zipped,self).__init__()
        self.size=size
    def forward(self, zipped, eps=1e-8, training=False):
        mus = [i[0] for i in zipped]
        logvars = [i[1] for i in zipped]
        mu,logvar = prior_expert(self.size,len(mus[0]))
        for i in range(len(mus)):
            mu=torch.cat((mu,mus[i].unsqueeze(0)),dim=0)
            logvar=torch.cat((logvar,logvars[i].unsqueeze(0)),dim=0)

        var       = torch.exp(logvar) + eps
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar
def prior_expert(size,batch_size):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1)
    """
    size   = (size[0],batch_size,size[2])
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    return mu.cuda(), logvar.cuda()
