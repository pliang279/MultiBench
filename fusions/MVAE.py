"""Implements MVAE."""
import torch
from torch import nn
from torch.autograd import Variable


class ProductOfExperts(nn.Module):
    """
    Return parameters for product of independent experts.
    
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """

    def __init__(self, size):
        """Initialize Product of Experts Object.

        Args:
            size (int): Size of Product of Experts Layer
        
        """
        super(ProductOfExperts, self).__init__()
        self.size = size

    def forward(self, mus, logvars, eps=1e-8):
        """Apply Product of Experts Layer.

        Args:
            mus (torch.Tensor): torch.Tensor of Mus
            logvars (torch.Tensor): torch.Tensor of Logvars
            eps (float, optional): Epsilon for log-exponent trick. Defaults to 1e-8.

        Returns:
            torch.Tensor, torch.Tensor: Output of PoE layer.
        """
        mu, logvar = _prior_expert(self.size, len(mus[0]))
        for i in range(len(mus)):
            
            mu = torch.cat((mu, mus[i].unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvars[i].unsqueeze(0)), dim=0)

        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


class ProductOfExperts_Zipped(nn.Module):
    """
    Return parameters for product of independent experts.
    
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """

    def __init__(self, size):
        """Initialize Product of Experts Object.

        Args:
            size (int): Size of Product of Experts Layer
        
        """
        super(ProductOfExperts_Zipped, self).__init__()
        self.size = size

    def forward(self, zipped, eps=1e-8):
        """Apply Product of Experts Layer.

        Args:
            mus (torch.Tensor): torch.Tensor of Mus
            logvars (torch.Tensor): torch.Tensor of Logvars
            eps (float, optional): Epsilon for log-exponent trick. Defaults to 1e-8.

        Returns:
            torch.Tensor, torch.Tensor: Output of PoE layer.
        """
        mus = [i[0] for i in zipped]
        logvars = [i[1] for i in zipped]
        mu, logvar = _prior_expert(self.size, len(mus[0]))
        for i in range(len(mus)):
            mu = torch.cat((mu, mus[i].unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, logvars[i].unsqueeze(0)), dim=0)

        var = torch.exp(logvar) + eps
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


def _prior_expert(size, batch_size):
    """
    Universal prior expert. Here we use a spherical.
    
    Gaussian: N(0, 1)
    """
    size = (size[0], batch_size, size[2])
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    return mu.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), logvar.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
