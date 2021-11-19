import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils.AUPRC import AUPRC


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, encoders, decoders, fuse):
        super(MVAE, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.fuse = fuse

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, inputs, training=False):
        mu, logvar = self.infer(inputs, training=training)
        
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)

        # reconstruct inputs based on that gaussian
        recons = []
        for i in range(len(inputs)):
            recons.append(self.decoders[i](z, training=training))
        return z, recons, mu, logvar

    def infer(self, inputs, training=False):
        
        mus = []
        logvars = []
        for i in range(len(inputs)):
            if inputs[i] is not None:
                mu, logvar = self.encoders[i](inputs[i], training=training)
                mus.append(mu)
                logvars.append(logvar)
        mu, logvar = self.fuse(mus, logvars)
        return mu, logvar


criterion = nn.CrossEntropyLoss()


def train_MVAE(encoders, decoders, head, fusion_method, train_dataloader, valid_dataloader, backbone_objective, total_epochs, finetune_epoch, finetune_every, backbone_optimtype=torch.optim.Adam,
               finetune_optimtype=torch.optim.SGD, finetune_batch_size=40, learning_rate=0.001, savedirbackbone='best1.pt', savedirhead='best2.pt'):
    def allnonebuti(i, item):
        ret = [None for w in encoders]
        ret[i] = item
        return ret
    n_modals = len(encoders)
    bestvalloss = 100
    mvae = MVAE(encoders, decoders, fusion_method)
    optim = backbone_optimtype(mvae.parameters(), lr=learning_rate)
    hoptim = finetune_optimtype(head.parameters(), lr=learning_rate)
    for ep in range(total_epochs):
        totalloss = 0.0
        totaltrain = 0
        for j in train_dataloader:
            optim.zero_grad()
            trains = [x.float().cuda() for x in j[:-1]]
            _, reconsjoint, mujoint, varjoint = mvae(trains, training=True)
            recons = []
            mus = []
            vars = []
            for i in range(len(trains)):
                _, recon, mu, var = mvae(
                    allnonebuti(i, trains[i]), training=True)
                recons.append(recon[i])
                mus.append(mu)
                vars.append(var)
            total_loss = backbone_objective(
                reconsjoint, trains, mujoint, varjoint)
            for i in range(len(trains)):
                total_loss += (backbone_objective(allnonebuti(i,
                               recons[i]), allnonebuti(i, trains[i]), mus[i], vars[i]))
            total_loss.backward()
            totalloss += total_loss*len(trains[0])
            totaltrain += len(trains[0])
            optim.step()
        print("epoch "+str(ep)+" train loss: "+str(totalloss/totaltrain))
        if ep > 0 and ep % finetune_every == 0:
            allpairs = []
            with torch.no_grad():
                for j in train_dataloader:
                    trains = [x.float().cuda() for x in j[:-1]]
                    _, recon, mu, var = mvae(trains, training=False)
                    for i in range(len(trains[0])):
                        allpairs.append((mu[i].cpu(), j[-1][i]))
            finetune_dataloader = DataLoader(
                allpairs, shuffle=True, num_workers=8, batch_size=finetune_batch_size)
            for eps in range(finetune_epoch):
                totalloss = 0.0
                for j in finetune_dataloader:
                    hoptim.zero_grad()
                    outs = head(j[0].cuda(), training=True)
                    loss = criterion(outs, j[-1].cuda())
                    loss.backward()
                    hoptim.step()
                    totalloss += loss * len(j[0])
                print("finetune epoch "+str(eps) +
                      " train loss: "+str(totalloss/totaltrain))
                with torch.no_grad():
                    totalloss = 0.0
                    total = 0
                    correct = 0
                    for j in valid_dataloader:
                        trains = [x.float().cuda() for x in j[:-1]]
                        _, mu, var = mvae(trains, training=False)
                        outs = head(mu, training=False)
                        loss = criterion(outs, j[-1].cuda())
                        totalloss += loss * len(j[0])
                        for i in range(len(j[-1])):
                            total += 1
                            if torch.argmax(outs[i]).item() == j[-1][i].item():
                                correct += 1
                    valoss = totalloss/total
                    print("valid loss: "+str(valoss) +
                          " acc: "+str(correct/total))
                    if valoss < bestvalloss:
                        bestvalloss = valoss
                        print('saving best')
                        torch.save(mvae, savedirbackbone)
                        torch.save(head, savedirhead)


def test_MVAE(mvae, head, test_dataloader, auprc=False):
    total = 0
    correct = 0
    pts = []
    for j in test_dataloader:
        xes = [x.float().cuda() for x in j[:-1]]
        y_batch = j[-1].cuda()
        with torch.no_grad():
            _, _, mu, var = mvae(xes, training=False)
            outs = head(mu, training=False)
            a = nn.Softmax()(outs)
        for ii in range(len(outs)):
            total += 1
            if outs[ii].tolist().index(max(outs[ii])) == y_batch[ii]:
                correct += 1
            pts.append([outs[ii][1], y_batch[ii]])

    print((float(correct)/total))
    print(AUPRC(pts))
