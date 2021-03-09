"""
To use: Call function train_MVAE with the following parameters for a datasets with m modalities m1, m2,... and num_classes kinds of labels

encoders: a list of m pytorch modules, the first one encodes input for m1, the second one encodes input for m2, etc. Its forward
function must output a tuple of two (mu,var), each of size (batch_size,n_latent)

decoders: a list of m pytorch modules, the first one decodes input for m1, the second one decodes input for m2, etc. Its forward
function must takes input of size (batch_size,n_latent) and outputs the same dimension as the inputs to the corresponding encoder.

head: nn module, the classification head to finetune, its forward function must take input of size (batch_size,n_latent)
and outputs (batch_size,num_classes)

trains: training data, as a list of m+1-tuples, each tuple contains the input tensor to m1, m2,... and the last element of the
tuple is the integer label

valids: valid data, same format as trains

batch_size: batch_size

n_latent: size of the latent representations of VAEs

modal_loss_funcs: a list of m reconstruction loss functions for each modality. Each must takes in 2 inputs of the dimension of its modality, and
outputs numerical loss for each element in the batch (not averaged or summed over the batch)

total_epochs: total number of epochs to run

finetune_epoch: how many finetune epochs to run for each finetune

finetune_every: finetunes every how many epochs

anneal: the annealing factor for KL divergence, default 0.0

learning_rate: learning rate, default 0.01

"""






import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.
  @param n_latents: integer
                    number of latent dimensions
  """

    def __init__(self, n_latents, encoders, decoders, batch_size):
        super(MVAE, self).__init__()
        self.bs = batch_size
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, inputs, training=True):
        mu, logvar = self.infer(inputs)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        # reconstruct inputs based on that gaussian
        recons = []
        for i in range(len(inputs)):
            recons.append(self.decoders[i](z));
        return recons, mu, logvar

    def infer(self, inputs):
        # print(inputs)
        batch_size = 0
        for i in inputs:
            if i is not None:
                batch_size = len(i)
                break

        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.n_latents))
        for i in range(len(inputs)):
            if inputs[i] is not None:
                # print(i)
                img_mu, img_logvar = self.encoders[i](inputs[i])
                mu = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
                logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
  See https://arxiv.org/pdf/1410.7827.pdf for equations.
  @param mu: M x D for M experts
  @param logvar: M x D for M experts
  """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


def prior_expert(size):
    """Universal prior expert. Here we use a spherical
  Gaussian: N(0, 1)
  """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    return mu.cuda(), logvar.cuda()


def elbo_loss(recons, origs, mu, logvar, modal_loss_funcs, weights, annealing=1.0):
    totalloss = 0.0
    if torch.max(logvar).item() > 99999:
        kld = logvar
    else:
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # print(origs)
    for i in range(len(recons)):
        if recons[i] is not None:
            # print(origs[i])
            totalloss += weights[i] * modal_loss_funcs[i](recons[i], origs[i])
    return torch.mean(totalloss + annealing * kld)


criterion = nn.CrossEntropyLoss()


def train_MVAE(encoders, decoders, head, train_datas, valid_datas, batch_size, n_latent, modal_loss_funcs, weights,
               total_epochs, finetune_epoch, finetune_every, anneal=0.0, learning_rate=0.01):
    def allnonebuti(i, item):
        ret = [None for w in weights]
        ret[i] = item
        return ret

    n_modals = len(encoders)
    bestvalloss = 100
    totaltrain = len(train_datas)
    totalval = len(valid_datas)
    mvae = MVAE(n_latent, encoders, decoders, batch_size)
    optim = torch.optim.Adam(mvae.parameters(), lr=learning_rate)
    hoptim = torch.optim.SGD(head.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_datas, shuffle=True, num_workers=8, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_datas, shuffle=False, num_workers=8, batch_size=batch_size)
    for ep in range(total_epochs):
        totalloss = 0.0
        for j in train_dataloader:
            optim.zero_grad()
            trains = [x.float().cuda() for x in j[:-1]]
            reconsjoint, mujoint, varjoint = mvae(trains, training=True)
            recons = []
            mus = []
            vars = []
            for i in range(len(trains)):
                recon, mu, var = mvae(allnonebuti(i, trains[i]), training=True)
                recons.append(recon[i])
                mus.append(mu)
                vars.append(var)
            total_loss = elbo_loss(reconsjoint, trains, mujoint, varjoint, modal_loss_funcs, weights, annealing=anneal)
            for i in range(len(trains)):
                # print(str(i)+" "+str(totalloss))
                # if i==1:
                # print(mus[i])
                # print(vars[i])
                total_loss += (elbo_loss(allnonebuti(i, recons[i]), allnonebuti(i, trains[i]), mus[i], vars[i],
                                         allnonebuti(i, modal_loss_funcs[i]), weights, annealing=anneal))
            total_loss.backward()
            totalloss += total_loss * len(trains[0])
            optim.step()
        print("epoch " + str(ep) + " train loss: " + str(totalloss / totaltrain))
        if ep > 0 and ep % finetune_every == 0:
            allpairs = []
            with torch.no_grad():
                for j in train_dataloader:
                    trains = [x.float().cuda() for x in j[:-1]]
                    recon, mu, var = mvae(trains, training=False)
                    for i in range(len(trains[0])):
                        allpairs.append((mu[i].cpu(), j[-1][i]))
            finetune_dataloader = DataLoader(allpairs, shuffle=True, num_workers=8, batch_size=batch_size)
            for eps in range(finetune_epoch):
                totalloss = 0.0
                for j in finetune_dataloader:
                    hoptim.zero_grad()
                    outs = head(j[0].cuda())
                    loss = criterion(outs, j[-1].cuda())
                    loss.backward()
                    hoptim.step()
                    totalloss += loss * len(j[0])
                print("finetune epoch " + str(eps) + " train loss: " + str(totalloss / totaltrain))
                with torch.no_grad():
                    totalloss = 0.0
                    total = 0
                    correct = 0
                    for j in valid_dataloader:
                        trains = [x.float().cuda() for x in j[:-1]]
                        _, mu, var = mvae(trains, training=False)
                        outs = head(mu)
                        loss = criterion(outs, j[-1].cuda())
                        totalloss += loss * len(j[0])
                        for i in range(len(j[-1])):
                            total += 1
                            if torch.argmax(outs[i]).item() == j[-1][i].item():
                                correct += 1
                    valoss = totalloss / totalval
                    print("valid loss: " + str(valoss) + " acc: " + str(correct / total))
                    if valoss < bestvalloss:
                        bestvalloss = valoss
                        print('saving best')
                        torch.save(mvae, 'best1.pt')
                        torch.save(head, 'best2.pt')







