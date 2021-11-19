from unimodals.common_models import MLP
import torch
import torchvision
import sys
import os
from memory_profiler import memory_usage
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm


def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


device = 2  # 1
batch_size = 16  # 64 # 5
num_workers = 1
lr = 1e-5  # 1e-4
sys.path.append(os.getcwd())

dataset_size = 'small'  # 'small', 'medium'
if dataset_size == 'small':
    num_classes = 5
else:
    num_classes = 20

# r3d = torchvision.models.video.r3d_18(pretrained=True)
# model=torch.nn.Sequential(r3d,MLP(400,200,5)).cuda(device)
r50 = torchvision.models.resnet50(pretrained=True)
r50.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model = torch.nn.Sequential(r50, MLP(1000, 200, 64),
                            torch.nn.Linear(64, num_classes)).cuda(device)
#odel=torch.load('best_kau_%s.pt' % dataset_size).cuda(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

print(getallparams([model]))

epochs = 15
if dataset_size == 'small':
    num_valid_loaders = 2
    num_train_loaders = 22
else:
    num_valid_loaders = 6
    num_train_loaders = 65


def train(ep=0):
    totalloss = 0.0
    total = 0
    model.train()
    for fid in range(num_train_loaders):
        print("epoch "+str(ep)+" subiter "+str(fid))
        if dataset_size == 'small':
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
        else:
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_medium/train/batch_medium'+str(fid)+'.pdt')
        datas = [d for d in datas if d[1].shape[1] == 763]
        train_dataloader = DataLoader(
            datas, shuffle=True, batch_size=batch_size)
        for j in train_dataloader:
            optim.zero_grad()
            x = j[1].unsqueeze(1)
            out = model(x.cuda(device))
            loss = criterion(out, j[2].cuda(device))
            loss.backward()
            optim.step()
            totalloss += loss*len(j[1])
            total += len(j[1])
    return totalloss/total

# mem = max(memory_usage(proc=train))



bestvaloss = 1000
for ep in tqdm(range(epochs)):
    train_loss = train(ep)
    print("Epoch "+str(ep)+" train loss: "+str(train_loss))
    model.eval()
    total = 0
    correct = 0
    totalloss = 0.0
    with torch.no_grad():
        for fid in range(num_valid_loaders):
            if dataset_size == 'small':
                datas = torch.load(
                    '/home/pliang/yiwei/kinetics_small/valid/batch_37%d.pdt' % fid)
            else:
                datas = torch.load(
                    '/home/pliang/yiwei/kinetics_medium/valid/batch_medium%d.pdt' % fid)
            valid_dataloader = DataLoader(
                datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
            for j in valid_dataloader:
                out = model(j[1].unsqueeze(1).cuda(device))
                loss = criterion(out, j[2].cuda(device))
                totalloss += loss*len(j[1])
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
        torch.save(model, 'best_kau_%s.pt' % dataset_size)

t0 = time.time()

print('testing')
if dataset_size == 'small':
    num_test_dataloaders = 3
else:
    num_test_dataloaders = 10
# model=torch.load('best_kau_%s.pt' % dataset_size).cuda(device)
valid_dataloader = None
total = 0
correct = 0
totalloss = 0.0
for fid in range(num_test_dataloaders):
    if dataset_size == 'small':
        datas = torch.load(
            '/home/pliang/yiwei/kinetics_small/test/batch_37%d.pdt' % fid)
    else:
        datas = torch.load(
            '/home/pliang/yiwei/kinetics_medium/test/batch_medium%d.pdt' % fid)
    test_dataloader = DataLoader(datas, shuffle=False, batch_size=batch_size)
    ys = []
    with torch.no_grad():
        for j in test_dataloader:
            out = model(j[1].unsqueeze(1).cuda(device))
            loss = criterion(out, j[2].cuda(device))
            totalloss += loss
            for ii in range(len(out)):
                ys.append(j[2][ii].item())
                total += 1
                if out[ii].tolist().index(max(out[ii])) == j[2][ii]:
                    correct += 1
print("Test loss: "+str(totalloss/total)+" acc: "+str(float(correct)/total))

t1 = time.time()
print(t1-t0)
