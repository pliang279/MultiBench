from unimodals.common_models import MLP
from fusions.common_fusions import Concat
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


device = 0  # 1
batch_size = 16  # 8 # 5
num_workers = 1  # 1
sys.path.append(os.getcwd())


class ResNetLSTMEnc(torch.nn.Module):
    def __init__(self, hiddim, dropout=False, dropoutp=0.1):
        super(ResNetLSTMEnc, self).__init__()
        self.enc = torchvision.models.resnet18(pretrained=True)
        self.lstm = nn.LSTM(1000, hiddim, batch_first=True)
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
        return hidden


class MMDL(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding

    def forward(self, inputs):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]], training=training))
        else:
            outs.append(self.encoders[0](inputs[0], training=training))
            outs.append(self.encoders[1](inputs[1].unsqueeze(1)))
        out = self.fuse(outs, training=training)
        
        return self.head(out, training=training)


r50 = torchvision.models.resnet50(pretrained=True)
r50.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
audio_model = torch.nn.Sequential(r50, MLP(1000, 200, 64))
encoders = [ResNetLSTMEnc(64).cuda(device), audio_model.cuda(device)]
fusion = Concat().cuda(device)
head = MLP(64+64, 200, 5).cuda(device)
model = MMDL(encoders, fusion, head, False).cuda(device)
# odel=torch.load('best_kslf.pt').cuda(device)
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
        print(len(datas))
        
        train_dataloader = DataLoader(
            datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        for j in train_dataloader:
            optim.zero_grad()
            out = model([i.float().cuda(device)
                        for i in j[:-1]], training=True)
            loss = criterion(out, j[2].cuda(device))
            loss.backward()
            optim.step()
            totalloss += loss*len(j[0])
            total += len(j[0])
    print("Epoch "+str(ep)+" train loss: "+str(totalloss/total))

# mem = max(memory_usage(proc=train))



num_data = 0
for fid in range(22):
    datas = torch.load(
        '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
    num_data += len(datas)
for fid in range(2):
    datas = torch.load(
        '/home/pliang/yiwei/kinetics_small/valid/batch_37%d.pdt' % fid)
    num_data += len(datas)
for fid in range(3):
    datas = torch.load(
        '/home/pliang/yiwei/kinetics_small/test/batch_37%d.pdt' % fid)
    num_data += len(datas)

'''
epochs = 15
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch_370.pdt')
valid_dataloader0 = DataLoader(datas,shuffle=False,batch_size=batch_size,num_workers=num_workers)
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch_371.pdt')
valid_dataloader1 = DataLoader(datas,shuffle=False,batch_size=batch_size,num_workers=num_workers)
valid_dataloaders = [valid_dataloader0, valid_dataloader1]
bestvaloss=1000
#a=input()
for ep in tqdm(range(epochs)):
    train(ep)
    model.eval()
    total = 0
    correct = 0
    totalloss = 0.0
    with torch.no_grad():
        for valid_dataloader in valid_dataloaders:
            for j in valid_dataloader:
                out = model([i.float().cuda(device) for i in j[:-1]],training=False)
                loss = criterion(out,j[2].cuda(device))
                totalloss += loss*len(j[0])
                for ii in range(len(out)):
                    total += 1
                    if out[ii].tolist().index(max(out[ii]))==j[2][ii]:
                        correct += 1
    valoss = totalloss/total
    print("Valid loss: "+str(totalloss/total)+" acc: "+str(float(correct)/total))
    if valoss < bestvaloss:
        print("Saving best")
        bestvaloss = valoss
        torch.save(model,'best_kslf.pt')
'''
t0 = time.time()

print('testing')
# model=torch.load('best_kslf.pt')
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
            out = model([i.float().cuda(device)
                        for i in j[:-1]], training=False)
            loss = criterion(out, j[2].cuda(device))
            totalloss += loss
            for ii in range(len(out)):
                total += 1
                if out[ii].tolist().index(max(out[ii])) == j[2][ii]:
                    correct += 1
print("Test loss: "+str(totalloss/total)+" acc: "+str(float(correct)/total))

t1 = time.time()
print(t1-t0)
