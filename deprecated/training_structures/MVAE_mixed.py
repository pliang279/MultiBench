import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils.AUPRC import AUPRC
from training_structures.MVAE_finetune import MVAE


criterion = nn.CrossEntropyLoss()


def train_MVAE(encoders, decoders, head, fusion_method, train_dataloader, valid_dataloader, backbone_objective, total_epochs, backbone_optimtype=torch.optim.Adam,
               finetune_optimtype=torch.optim.SGD, ce_weight=2.0, finetune_batch_size=40, learning_rate=0.001, savedirbackbone='best1.pt', savedirhead='best2.pt'):
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
            z, reconsjoint, mujoint, varjoint = mvae(trains, training=True)
            recons = []
            mus = []
            vars = []
            for i in range(len(trains)):
                _, recon, mu, var = mvae(
                    allnonebuti(i, trains[i]), training=True)
                recons.append(recon[i])
                mus.append(mu)
                vars.append(var)
                
                
            # exit(0)
            total_loss = backbone_objective(
                reconsjoint, trains, mujoint, varjoint)
            

            for i in range(len(trains)):
                total_loss += (backbone_objective(allnonebuti(i,
                                                              recons[i]), allnonebuti(i, trains[i]), mus[i], vars[i]))
            ceout = head(z)
            celoss = criterion(ceout, j[-1].cuda())
            total_loss += celoss*ce_weight
            total_loss.backward()
            totalloss += total_loss*len(trains[0])
            totaltrain += len(trains[0])
            optim.step()
        print("epoch "+str(ep)+" train loss: "+str(totalloss/totaltrain))
        if True:
            with torch.no_grad():
                totalloss = 0.0
                total = 0
                correct = 0
                for j in valid_dataloader:
                    trains = [x.float().cuda() for x in j[:-1]]
                    _, _, mu, var = mvae(trains, training=False)
                    outs = head(mu)
                    loss = criterion(outs, j[-1].cuda())
                    totalloss += loss * len(j[0])
                    for i in range(len(j[-1])):
                        total += 1
                        if torch.argmax(outs[i]).item() == j[-1][i].item():
                            correct += 1
                valoss = totalloss/total
                print("valid loss: "+str(valoss)+" acc: "+str(correct/total))
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
            pts.append([a[ii][1], y_batch[ii]])

    print((float(correct)/total))
    if(auprc):
        print(AUPRC(pts))

    return float(correct)/total
