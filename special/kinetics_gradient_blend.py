from unimodals.common_models import MLP
from fusions.common_fusions import Concat
import copy
import torch
import torch.nn as nn
import torchvision
import sys
import os
import time
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


device = 1
batch_size = 16
num_workers = 1
sys.path.append(os.getcwd())


def multimodalcompute(models, train_x):
    outs = []
    for i in range(len(models)):
        outs.append(models[i](train_x[i]))
    return outs


def multimodalcondense(models, fuse, train_x):
    outs = multimodalcompute(models, train_x)
    return fuse(outs)


def gettrainloss(model, head, monum, batch_size, num_workers):
    losses = 0.0
    total = 0
    with torch.no_grad():
        for fid in range(22):
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
            train_dataloader = DataLoader(
                datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            for j in train_dataloader:
                total += len(j[0])
                train_x = j[monum].float().cuda(device)
                if monum == 1:
                    train_x = train_x.unsqueeze(1)
                train_y = j[-1].cuda(device)
                out = model(train_x)
                out = head(out)
                loss = criterion(out, train_y)
                losses += loss*len(j[0])
    return losses/total


def getvalloss(model, head, loaders, monum):
    losses = 0.0
    total = 0
    with torch.no_grad():
        for loader in loaders:
            for j in loader:
                total += len(j[0])
                train_x = j[monum].float().cuda(device)
                if monum == 1:
                    train_x = train_x.unsqueeze(1)
                train_y = j[-1].cuda(device)
                out = model(train_x)
                out = head(out)
                loss = criterion(out, train_y)
                losses += loss*len(j[0])
    return losses/total


def gettrainmloss(models, head, fuse, batch_size, num_workers):
    losses = 0.0
    total = 0
    with torch.no_grad():
        for fid in range(22):
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
            train_dataloader = DataLoader(
                datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            for j in train_dataloader:
                total += len(j[0])
                train_x = [j[0].float().cuda(
                    device), j[1].unsqueeze(1).float().cuda(device)]
                train_y = j[-1].cuda(device)
                out = head(multimodalcondense(models, fuse, train_x))
                loss = criterion(out, train_y)
                losses += loss*len(j[0])
    return losses/total


def getvalmloss(models, head, fuse, loaders):
    losses = 0.0
    total = 0
    with torch.no_grad():
        for loader in loaders:
            for j in loader:
                total += len(j[0])
                train_x = [j[0].float().cuda(
                    device), j[1].unsqueeze(1).float().cuda(device)]
                train_y = j[-1].cuda(device)
                out = head(multimodalcondense(models, fuse, train_x))
                loss = criterion(out, train_y)
                losses += loss*len(j[0])
    return losses/total


class completeModule(nn.Module):
    def __init__(self, encoders, fuse, head):
        super(completeModule, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fuse
        self.head = head

    def forward(self, x, training=False):
        outs = multimodalcondense(self.encoders, self.fuse, x)
        return self.head(outs, training=training)


class ResNetLSTMEnc(torch.nn.Module):
    def __init__(self, hiddim, dropout=False, dropoutp=0.1):
        super(ResNetLSTMEnc, self).__init__()
        self.enc = torchvision.models.resnet18(pretrained=True)
        self.lstm = nn.LSTM(1000, hiddim, batch_first=True)
        self.dropoutp = dropoutp
        self.dropout = dropout

    def forward(self, x, training=True):  # x is (cbatch_size, 3, 150, 112, 112)
        cbatch_size = x.shape[0]
        x = x.permute([0, 2, 1, 3, 4])  # (cbatch_size, 150, 3, 112, 112)
        x = x.reshape(-1, 3, 112, 112)  # (cbatch_size*150, 3, 112, 112)
        x = self.enc(x)  # (cbatch_size*150, 1000)
        x = x.reshape(cbatch_size, -1, 1000)
        hidden = self.lstm(x)[1][0]
        hidden = hidden.permute([1, 2, 0])
        hidden = hidden.reshape([hidden.size()[0], -1])
        if self.dropout:
            hidden = F.dropout(hidden, p=self.dropoutp, training=training)
        return hidden


class MMDL(nn.Module):
    def __init__(self, encoders, fusion, head, has_padding=False):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding

    def forward(self, inputs, training=False):
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


criterion = torch.nn.CrossEntropyLoss()

r50 = torchvision.models.resnet50(pretrained=True)
r50.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
audio_model = torch.nn.Sequential(r50, MLP(1000, 200, 64))
unimodal_models = [ResNetLSTMEnc(64).cuda(
    device), audio_model.cuda(device)]  # encoders
fuse = Concat().cuda(device)  # fusion
multimodal_classification_head = MLP(128, 200, 5).cuda(device)
unimodal_classification_heads = [
    MLP(64, 200, 5).cuda(device), MLP(64, 200, 5).cuda(device)]

lr = 0.0001
params = []
for model in unimodal_models:
    params.extend(model.parameters())
for model in unimodal_classification_heads:
    params.extend(model.parameters())
params.extend(multimodal_classification_head.parameters())
params.extend(fuse.parameters())
optim = torch.optim.Adam(params, lr=lr)

finetunehead = copy.deepcopy(multimodal_classification_head).cuda(device)
fusehead = copy.deepcopy(fuse).cuda(device)
params = list(finetunehead.parameters())
if fuse.parameters() is not None:
    params.extend(list(fuse.parameters()))
optimi = torch.optim.Adam(params, lr=lr)
bestvalloss = 10000.0

num_epoch = 60  # 30 # 16
gb_epoch = 6  # 3 # 2
finetune_epoch = 3  # 2
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch_370.pdt')
valid_dataloader0 = DataLoader(
    datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch_371.pdt')
valid_dataloader1 = DataLoader(
    datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
valid_dataloaders = [valid_dataloader0, valid_dataloader1]
delta = False
bestvaloss = 1000.0

for ep in tqdm(range(num_epoch//gb_epoch)):
    # gb_estimate
    weights = []
    for monum in range(len(unimodal_models)):
        print("At gb_estimate unimodal "+str(monum))
        model = copy.deepcopy(unimodal_models[monum]).cuda(device)
        head = copy.deepcopy(unimodal_classification_heads[monum]).cuda(device)
        optim = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=lr)
        ltN = gettrainloss(model, head, monum, batch_size, num_workers)
        lvN = getvalloss(model, head, valid_dataloaders, monum)
        for i in range(gb_epoch):
            totalloss = 0.0
            total = 0
            for fid in range(22):
                datas = torch.load(
                    '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
                train_dataloader = DataLoader(
                    datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
                for j in train_dataloader:
                    total += len(j[0])
                    train_x = j[monum].float().cuda(device)
                    if monum == 1:
                        train_x = train_x.unsqueeze(1)
                    train_y = j[-1].cuda(device)
                    optim.zero_grad()
                    out = model(train_x)
                    out = head(out)
                    loss = criterion(out, train_y)
                    totalloss += loss * len(j[0])
                    loss.backward()
                    optim.step()
            print("Epoch "+str(i)+" loss: "+str(totalloss / total))
        ltNn = gettrainloss(model, head, monum, batch_size, num_workers)
        lvNn = getvalloss(model, head, valid_dataloaders, monum)
        print("Final train loss: "+str(ltNn)+" valid loss: "+str(lvNn))
        oNn = lvNn-ltNn
        oN = lvN-ltN
        if delta:
            oi = oNn-oN
            g = lvNn-lvN
        else:
            oi = oNn
            if oi < 0:
                oi = 0.0001
            g = lvNn
        print("raw: "+str(g/(oi*oi)))
        w = abs(g/(oi*oi))
        weights.append(w)
    print("At gb_estimate multimodal ")
    allcopies = [copy.deepcopy(x).cuda(device) for x in unimodal_models]
    mmcopy = copy.deepcopy(multimodal_classification_head).cuda(device)
    fusecopy = copy.deepcopy(fuse).cuda(device)
    params = []
    for model in allcopies:
        params.extend(list(model.parameters()))
    params.extend(list(mmcopy.parameters()))
    if fusecopy.parameters() is not None:
        params.extend(list(fusecopy.parameters()))
    optim = torch.optim.Adam(params, lr=lr)
    ltN = gettrainmloss(allcopies, mmcopy, fusecopy, batch_size, num_workers)
    lvN = getvalmloss(allcopies, mmcopy, fusecopy, valid_dataloaders)
    for i in range(gb_epoch):
        totalloss = 0.0
        total = 0
        for fid in range(22):
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
            train_dataloader = DataLoader(
                datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            for j in train_dataloader:
                total += len(j[0])
                # train_x = [x.float().cuda(device) for x in j[:-1]]
                train_x = [j[0].float().cuda(
                    device), j[1].unsqueeze(1).float().cuda(device)]
                train_y = j[-1].cuda(device)
                optim.zero_grad()
                out = mmcopy(multimodalcondense(allcopies, fusecopy, train_x))
                loss = criterion(out, train_y)
                totalloss += loss*len(j[0])
                loss.backward()
                optim.step()
        print("Epoch "+str(i)+" loss: "+str(totalloss/total))
    ltNn = gettrainmloss(allcopies, mmcopy, fusecopy, batch_size, num_workers)
    lvNn = getvalmloss(allcopies, mmcopy, fusecopy, valid_dataloaders)
    print("Final train loss: "+str(ltNn)+" valid loss: "+str(lvNn))
    oNn = lvNn-ltNn
    oN = lvN-ltN
    if delta:
        oi = oNn-oN
        g = lvNn-lvN
    else:
        oi = oNn
        if oi < 0:
            oi = 0.0001
        g = lvNn
    print("raw: "+str(g/(oi*oi)))
    mw = abs(g/(oi*oi))
    weights.append(mw)
    z = sum(weights)
    weights = [(w/z).item() for w in weights]

    print("epoch "+str(ep*gb_epoch)+" weights: "+str(weights))

    # multimodal
    for jj in range(gb_epoch):
        totalloss = 0.0
        total = 0
        for fid in range(22):
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
            train_dataloader = DataLoader(
                datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            for j in train_dataloader:
                train_x = [j[0].float().cuda(
                    device), j[1].unsqueeze(1).float().cuda(device)]
                train_y = j[2].cuda(device)
                optim.zero_grad()
                outs = multimodalcompute(unimodal_models, train_x)
                catout = fuse(outs, training=True)
                blendloss = criterion(multimodal_classification_head(
                    catout, training=True), train_y)*weights[-1]
                for ii in range(len(unimodal_models)):
                    loss = criterion(unimodal_classification_heads[ii](
                        outs[ii]), train_y)
                    blendloss += loss * weights[ii]
                totalloss += blendloss*len(j[0])
                blendloss.backward()
                optim.step()
                total += len(j[0])
        print("epoch "+str(jj+ep*gb_epoch)+" blend train loss: " +
              str(totalloss/total))

    # finetunes classification head
    finetunetrains = []
    with torch.no_grad():
        for fid in range(22):
            datas = torch.load(
                '/home/pliang/yiwei/kinetics_small/train/batch_37'+str(fid)+'.pdt')
            train_dataloader = DataLoader(
                datas, shuffle=True, batch_size=batch_size, num_workers=num_workers)
            for j in train_dataloader:
                # train_x = [x.float().cuda(device) for x in j[:-1]]
                train_x = [j[0].float().cuda(
                    device), j[1].unsqueeze(1).float().cuda(device)]
                train_y = j[2].cuda(device)
                outs = multimodalcompute(unimodal_models, train_x)
                for iii in range(len(train_y)):
                    aa = [x[iii].cpu() for x in outs]
                    aa.append(train_y[iii].cpu())
                    finetunetrains.append(aa)
    print("Length of ftt_dataloader: "+str(len(finetunetrains)))
    ftt_dataloader = DataLoader(
        finetunetrains, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    for jj in range(finetune_epoch):
        totalloss = 0.0
        for j in ftt_dataloader:
            optimi.zero_grad()
            train_x = [j[0].float().cuda(
                device), j[1].unsqueeze(1).float().cuda(device)]
            train_y = j[-1].cuda(device)
            blendloss = criterion(finetunehead(
                fusehead(train_x, training=True), training=True), train_y)
            totalloss += blendloss * len(j[0])
            blendloss.backward()
            optimi.step()
        print("finetune train loss: "+str(totalloss/len(finetunetrains)))
        with torch.no_grad():
            totalloss = 0.0
            total = 0
            corrects = 0
            for valid_dataloader in valid_dataloaders:
                for j in valid_dataloader:
                    valid_x = [j[0].float().cuda(
                        device), j[1].unsqueeze(1).float().cuda(device)]
                    valid_y = j[-1].cuda(device)
                    outs = multimodalcompute(unimodal_models, valid_x)
                    catout = fusehead(outs, training=False)
                    predicts = finetunehead(catout, training=False)
                    blendloss = criterion(predicts, valid_y)
                    totalloss += blendloss*len(j[0])
                    predictlist = predicts.tolist()
                    for ii in range(len(j[0])):
                        total += 1
                        if predictlist[ii].index(max(predictlist[ii])) == valid_y[ii]:
                            corrects += 1
            valoss = totalloss/total
            print("epoch "+str((ep+1)*gb_epoch-1)+" valid loss: "+str(totalloss/total) +
                  (" acc: "+str(float(corrects)/total)))
            if valoss < bestvaloss:
                print("Saving best")
                bestvaloss = valoss
                torch.save(completeModule(unimodal_models,
                           fusehead, finetunehead), 'best_kgrb.pt')

print('testing')
model = torch.load('best_kgrb.pt').cuda(device)
valid_dataloader = None
total = 0
corrects = 0
totalloss = 0.0
for fid in range(3):
    datas = torch.load(
        '/home/pliang/yiwei/kinetics_small/test/batch_37%d.pdt' % fid)
    test_dataloader = DataLoader(
        datas, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        for j in test_dataloader:
            valid_x = [j[0].float().cuda(
                device), j[1].unsqueeze(1).float().cuda(device)]
            valid_y = j[-1].cuda(device)
            predicts = model(valid_x)
            blendloss = criterion(predicts, valid_y.squeeze())
            totalloss += blendloss*len(j[0])
            predictlist = predicts.tolist()
            for ii in range(len(j[0])):
                total += 1
                if predictlist[ii].index(max(predictlist[ii])) == valid_y[ii]:
                    corrects += 1
print("Test loss: "+str(totalloss/total)+" acc: "+str(float(corrects)/total))
