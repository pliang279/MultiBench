import torch
import torch.nn as nn
import torchvision
import sys
from memory_profiler import memory_usage
import os
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


device = 0
batch_size = 16  # 8 # 5
num_workers = 1  # 1
sys.path.append(os.getcwd())


class ResNetLSTM(torch.nn.Module):
    def __init__(self, hiddim, outdim, dropout=False, dropoutp=0.1):
        super(ResNetLSTM, self).__init__()
        self.enc = torchvision.models.resnet18(pretrained=True)
        self.lstm = nn.LSTM(1000, hiddim, batch_first=True)
        self.linear = nn.Linear(hiddim, outdim)
        self.dropoutp = dropoutp
        self.dropout = dropout

    def forward(self, x):  # x is (cbatch_size, 3, 150, 112, 112)
        cbatch_size = x.shape[0]
        x = x.permute([0, 2, 1, 3, 4])  # (cbatch_size, 150, 3, 112, 112)
        x = x.reshape(-1, 3, 112, 112)  # (cbatch_size*150, 3, 112, 112)
        x = self.enc(x)  # (cbatch_size*150, 1000)
        x = x.reshape(cbatch_size, -1, 1000)
        hidden = self.lstm(x)[1][0]
        hidden = hidden.permute([1, 2, 0])
        hidden = hidden.reshape([hidden.size()[0], -1])
        if self.dropout:
            hidden = F.dropout(hidden, p=self.dropoutp)
        out = self.linear(hidden)
        return out


model = ResNetLSTM(64, 5).cuda(device)
# model=torch.load('best_kvu.pt').cuda(device)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

print(getallparams([model]))


def train(ep=0):
    totalloss = 0.0
    total = 0
    model.train()
    for fid in range(22):
        print("epoch "+str(ep)+" subiter "+str(fid))
        datas = torch.load(
            '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
        
        train_dataloader = DataLoader(
            datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        for j in train_dataloader:
            optim.zero_grad()
            model.train()
            out = model(j[0].cuda(device))
            loss = criterion(out, j[2].cuda(device))
            loss.backward()
            optim.step()
            totalloss += loss*len(j[0])
            total += len(j[0])
    print("Epoch "+str(ep)+" train loss: "+str(totalloss/total))

# mem = max(memory_usage(proc=train))



epochs = 15
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch_370.pdt')
valid_dataloader0 = DataLoader(
    datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch_371.pdt')
valid_dataloader1 = DataLoader(
    datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
valid_dataloaders = [valid_dataloader0, valid_dataloader1]
bestvaloss = 1000
# a=input()
for ep in tqdm(range(epochs)):
    train(ep)
    model.eval()
    total = 0
    correct = 0
    totalloss = 0.0
    with torch.no_grad():
        for valid_dataloader in valid_dataloaders:
            for j in valid_dataloader:
                model.train()
                out = model(j[0].cuda(device))
                loss = criterion(out, j[2].cuda(device))
                totalloss += loss*len(j[0])
                for ii in range(len(out)):
                    total += 1
                    if out[ii].tolist().index(max(out[ii])) == j[2][ii]:
                        correct += 1
    valoss = totalloss/total
    print("Valid loss: "+str(totalloss/total) +
          " acc: "+str(float(correct)/total))
    if valoss < bestvaloss:
        print("Saving best")
        bestvaloss = valoss
        torch.save(model, 'best_kvu.pt')
t0 = time.time()

print('testing')
model = torch.load('best_kvu.pt')
valid_dataloader = None
total = 0
correct = 0
totalloss = 0.0
for fid in range(3):
    datas = torch.load(
        '/home/pliang/yiwei/kinetics_small/test/batch_37%d.pdt' % fid)
    test_dataloader = DataLoader(
        datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for j in test_dataloader:
            model.eval()
            out = model(j[0].cuda(device))
            loss = criterion(out, j[2].cuda(device))
            totalloss += loss
            for ii in range(len(out)):
                total += 1
                if out[ii].tolist().index(max(out[ii])) == j[2][ii]:
                    correct += 1
print("Test loss: "+str(totalloss/total)+" acc: "+str(float(correct)/total))

t1 = time.time()
print(t1-t0)
