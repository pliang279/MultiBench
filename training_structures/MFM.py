import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
from utils.AUPRC import AUPRC


class MFM(nn.Module):
    def __init__(self, encoders, decoders, fusionmodule, head, intermediates):
        super(MFM, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.intermediates = nn.ModuleList(intermediates)
        self.fuse = fusionmodule
        self.head = head

    def forward(self, inputs, training=True):
        outs = []
        for i in range(len(inputs)):
          outs.append(self.encoders[i](inputs[i]))
        #print(outs[0].size())
        #print(outs[1].size())
        fused = self.fuse(outs)
        #print(fused.size())
        combined = self.intermediates[-1](fused)
        recons = []
        for i in range(len(outs)):
          outs[i] = self.intermediates[i](outs[i])
        for i in range(len(inputs)):
          recons.append(self.decoders[i](
              torch.cat([outs[i], combined], dim=1)))
        return recons, self.head(combined)


criterion = nn.CrossEntropyLoss()


def train_MFM(encoders, decoders, head, intermediates, fusion, recon_loss_func, train_dataloader, valid_dataloader, total_epochs, ce_weight=2.0, learning_rate=0.001, savedir='best.pt'):
  n_modals = len(encoders)
  bestvalloss = 100
  mvae = MFM(encoders, decoders, fusion, head, intermediates)
  optim = torch.optim.Adam(mvae.parameters(), lr=learning_rate)
  for ep in range(total_epochs):
    totalloss = 0.0
    total = 0
    for j in train_dataloader:
      optim.zero_grad()
      trains = [x.float().cuda() for x in j[:-1]]
      total += len(trains[0])
      recons, outs = mvae(trains, training=True)
      loss = criterion(outs, j[-1].cuda())*ce_weight
      #print(loss)
      loss += recon_loss_func(recons, trains)
      #print(loss)
      loss.backward()
      totalloss += loss*len(trains[0])
      optim.step()
    print("epoch "+str(ep)+" train loss: "+str(totalloss/total))
    if True:
      with torch.no_grad():
        totalloss = 0.0
        total = 0
        correct = 0
        for j in valid_dataloader:
          trains = [x.float().cuda() for x in j[:-1]]
          _, outs = mvae(trains, training=False)
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
          torch.save(mvae, savedir)


def test_MFM(model, test_dataloader, auprc=False):
  total = 0
  correct = 0
  batchsize = 40
  pts = []
  for j in test_dataloader:
    xes = [x.float().cuda() for x in j[:-1]]
    y_batch = j[-1].cuda()
    with torch.no_grad():
      _, outs = model(xes, training=False)
      a = nn.Softmax()(outs)
    for ii in range(len(outs)):
      total += 1
      if outs[ii].tolist().index(max(outs[ii])) == y_batch[ii]:
        correct += 1
      pts.append([a[ii][1], y_batch[ii]])
  print((float(correct)/total))
  if auprc:
    print(AUPRC(pts))
  return float(correct)/total
