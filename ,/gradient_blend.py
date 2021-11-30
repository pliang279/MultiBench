import sklearn.metrics
import torch
from torch import nn
import copy
import random
from torch.utils.data import DataLoader, Subset

criterion = nn.CrossEntropyLoss()
delta = False


def getloss(model, head, data, monum, batch_size):
    losses = 0.0
    total = 0
    with torch.no_grad():
        for j in data:
            total += len(j[0])
            train_x = j[monum].float().cuda()
            train_y = j[-1].cuda()
            out = model(train_x)
            # if (monum==1):
            
            out = head(out)
            loss = criterion(out, train_y.squeeze())
            
            losses += loss*len(j[0])
    return losses/total


def train_unimodal(model, head, optim, trains, valids, monum, epoch, batch_size):

    ltN = getloss(model, head, trains, monum, batch_size)
    lvN = getloss(model, head, valids, monum, batch_size)
    for i in range(epoch):
        totalloss = 0.0
        total = 0
        for j in trains:
            total += len(j[0])
            train_x = j[monum].float().cuda()
            train_y = j[-1].cuda()
            optim.zero_grad()
            out = model(train_x)
            out = head(out)
            loss = criterion(out, train_y.squeeze())
            totalloss += loss * len(j[0])
            loss.backward()
            optim.step()
        print("Epoch "+str(i)+" loss: "+str(totalloss / total))

    ltNn = getloss(model, head, trains, monum, batch_size)
    lvNn = getloss(model, head, valids, monum, batch_size)
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
    return abs(g/(oi*oi))


def multimodalcondense(models, fuse, train_x):
    outs = multimodalcompute(models, train_x)
    return fuse(outs)


def multimodalcompute(models, train_x):
    outs = []
    for i in range(len(models)):
        outs.append(models[i](train_x[i]))
    return outs


def getmloss(models, head, fuse, data, batch_size):
    losses = 0.0
    total = 0
    with torch.no_grad():
        for j in data:
            total += len(j[0])
            train_x = [x.float().cuda() for x in j[:-1]]
            train_y = j[-1].cuda()
            out = head(multimodalcondense(models, fuse, train_x))
            loss = criterion(out, train_y.squeeze())
            losses += loss*len(j[0])
    return losses/float(total)


def train_multimodal(models, head, fuse, optim, trains, valids, epoch, batch_size):
    ltN = getmloss(models, head, fuse, trains, batch_size)
    lvN = getmloss(models, head, fuse, valids, batch_size)
    for i in range(epoch):
        totalloss = 0.0
        total = 0
        for j in trains:
            total += len(j[0])
            train_x = [x.float().cuda() for x in j[:-1]]
            train_y = j[-1].cuda()
            optim.zero_grad()
            out = head(multimodalcondense(models, fuse, train_x))
            loss = criterion(out, train_y.squeeze())
            totalloss += loss*len(j[0])
            loss.backward()
            optim.step()
        print("Epoch "+str(i)+" loss: "+str(totalloss/total))
    ltNn = getmloss(models, head, fuse, trains, batch_size)
    lvNn = getmloss(models, head, fuse, valids, batch_size)
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
    return abs(g/(oi*oi))


def gb_estimate(unimodal_models, multimodal_classification_head, fuse, unimodal_classification_heads, train_dataloader, gb_epoch,
                batch_size, v_dataloader, lr, weight_decay=0.0, optimtype=torch.optim.SGD):
    weights = []
    for i in range(len(unimodal_models)):
        print("At gb_estimate unimodal "+str(i))
        model = copy.deepcopy(unimodal_models[i]).cuda()
        head = copy.deepcopy(unimodal_classification_heads[i]).cuda()
        optim = optimtype(list(model.parameters()) +
                          list(head.parameters()), lr=lr, weight_decay=weight_decay)
        w = train_unimodal(model, head, optim, train_dataloader,
                           v_dataloader, i, gb_epoch, batch_size)
        weights.append(w)
    print("At gb_estimate multimodal ")
    allcopies = [copy.deepcopy(x).cuda() for x in unimodal_models]
    mmcopy = copy.deepcopy(multimodal_classification_head).cuda()
    fusecopy = copy.deepcopy(fuse).cuda()
    params = []
    for model in allcopies:
        params.extend(list(model.parameters()))
    params.extend(list(mmcopy.parameters()))
    if fusecopy.parameters() is not None:
        params.extend(list(fusecopy.parameters()))
    optim = optimtype(params, lr=lr, weight_decay=weight_decay)
    weights.append(train_multimodal(allcopies, mmcopy, fusecopy,
                                    optim, train_dataloader, v_dataloader, gb_epoch, batch_size))
    z = sum(weights)
    return [(w/z).item() for w in weights]


softmax = nn.Softmax()


class completeModule(nn.Module):
    def __init__(self, encoders, fuse, head):
        super(completeModule, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fuse
        self.head = head

    def forward(self, x):
        outs = multimodalcondense(self.encoders, self.fuse, x)
        return self.head(outs)


def calcAUPRC(pts):
    true_labels = [int(x[1]) for x in pts]
    predicted_probs = [x[0] for x in pts]
    return sklearn.metrics.average_precision_score(true_labels, predicted_probs)


def train(unimodal_models,  multimodal_classification_head,
          unimodal_classification_heads, fuse, train_dataloader, valid_dataloader,
          num_epoch, lr, gb_epoch=20, v_rate=0.08, weight_decay=0.0, optimtype=torch.optim.SGD,
          finetune_epoch=25, classification=True, AUPRC=False, savedir='best.pt'):
    params = []
    for model in unimodal_models:
        params.extend(model.parameters())
    for model in unimodal_classification_heads:
        params.extend(model.parameters())
    params.extend(multimodal_classification_head.parameters())
    params.extend(fuse.parameters())
    optim = optimtype(params, lr=lr, weight_decay=weight_decay)
    train_datas = train_dataloader.dataset
    splitloc = int(len(train_datas)*v_rate)
    inds = list(range(len(train_datas)))
    train_inds = inds[splitloc:]
    v_inds = inds[0:splitloc]
    # train_data = train_datas[splitloc:]
    # v_data = train_datas[0:splitloc]
    train_data = Subset(train_datas, train_inds)
    v_data = Subset(train_datas, v_inds)
    train_dataloader = DataLoader(
        train_data, shuffle=True, num_workers=8, batch_size=train_dataloader.batch_size)
    tv_dataloader = DataLoader(
        v_data, shuffle=False, num_workers=8, batch_size=train_dataloader.batch_size)
    finetunehead = copy.deepcopy(multimodal_classification_head).cuda()
    fusehead = copy.deepcopy(fuse).cuda()
    params = list(finetunehead.parameters())
    if fuse.parameters() is not None:
        params.extend(list(fuse.parameters()))
    optimi = optimtype(params, lr=lr, weight_decay=weight_decay)
    bestvalloss = 10000.0
    for i in range(num_epoch//gb_epoch):
        # """
        weights = gb_estimate(unimodal_models,  multimodal_classification_head, fuse,
                              unimodal_classification_heads, train_dataloader, gb_epoch, train_dataloader.batch_size, tv_dataloader, lr, weight_decay, optimtype)
        # """
        # weights=(1.0,1.0,1.0)
        print("epoch "+str(i*gb_epoch)+" weights: "+str(weights))
        for jj in range(gb_epoch):
            totalloss = 0.0
            for j in train_dataloader:
                train_x = [x.float().cuda() for x in j[:-1]]
                train_y = j[-1].cuda()
                optim.zero_grad()
                outs = multimodalcompute(unimodal_models, train_x)
                fuse.train()
                multimodal_classification_head.train()
                catout = fuse(outs)
                blendloss = criterion(multimodal_classification_head(
                    catout), train_y.squeeze())*weights[-1]
                for ii in range(len(unimodal_models)):
                    loss = criterion(unimodal_classification_heads[ii](
                        outs[ii]), train_y.squeeze())
                    blendloss += loss * weights[ii]
                totalloss += blendloss*len(j[0])
                blendloss.backward()
                optim.step()
            print("epoch "+str(jj+i*gb_epoch)+" blend train loss: " +
                  str(totalloss/len(train_data)))
        # finetunes classification head
        finetunetrains = []
        with torch.no_grad():
            for j in train_dataloader:
                train_x = [x.float().cuda() for x in j[:-1]]
                train_y = j[-1].cuda()
                outs = multimodalcompute(unimodal_models, train_x)
                for iii in range(len(train_y)):
                    aa = [x[iii].cpu() for x in outs]
                    
                    aa.append(train_y[iii].cpu())
                    
                    finetunetrains.append(aa)
        print("Length of ftt_dataloader: "+str(len(finetunetrains)))
        ftt_dataloader = DataLoader(
            finetunetrains, shuffle=True, num_workers=8, batch_size=train_dataloader.batch_size)
        for jj in range(finetune_epoch):
            totalloss = 0.0
            for j in ftt_dataloader:
                optimi.zero_grad()
                train_x = [x.float().cuda() for x in j[:-1]]
                train_y = j[-1].cuda()
                finetunehead.train()
                fusehead.train()
                blendloss = criterion(finetunehead(
                    fusehead(train_x)), train_y.squeeze())
                totalloss += blendloss * len(j[0])
                blendloss.backward()
                optimi.step()
            print("finetune train loss: "+str(totalloss/len(train_data)))
            with torch.no_grad():
                totalloss = 0.0
                total = 0
                corrects = 0
                auprclist = []
                for j in valid_dataloader:
                    valid_x = [x.float().cuda() for x in j[:-1]]
                    valid_y = j[-1].cuda()
                    outs = multimodalcompute(unimodal_models, valid_x)
                    fusehead.eval()
                    finetunehead.eval()
                    catout = fusehead(outs)
                    predicts = finetunehead(catout)
                    blendloss = criterion(predicts, valid_y.squeeze())
                    totalloss += blendloss*len(j[0])
                    predictlist = predicts.tolist()
                    for ii in range(len(j[0])):
                        total += 1
                        if AUPRC:
                            predictval = softmax(predicts[ii])
                            auprclist.append(
                                (predictval[1].item(), valid_y[ii].item()))
                        if classification:
                            if predictlist[ii].index(max(predictlist[ii])) == valid_y[ii]:
                                corrects += 1
                valoss = totalloss/total
                print("epoch "+str((i+1)*gb_epoch-1)+" valid loss: "+str(totalloss/total) +
                      ((" acc: "+str(float(corrects)/total)) if classification else ''))
                if AUPRC:
                    print("With AUPRC: "+str(calcAUPRC(auprclist)))
                if valoss < bestvalloss:
                    bestvalloss = valoss
                    print("Saving best")
                    torch.save(completeModule(unimodal_models,
                                              fusehead, finetunehead), savedir)


def test(model, test_dataloader, auprc=False, classification=True):
    with torch.no_grad():
        totalloss = 0.0
        total = 0
        corrects = 0
        auprclist = []
        for j in test_dataloader:
            valid_x = [x.float().cuda() for x in j[:-1]]
            valid_y = j[-1].cuda()
            predicts = model(valid_x)
            blendloss = criterion(predicts, valid_y.squeeze())
            totalloss += blendloss*len(j[0])
            predictlist = predicts.tolist()
            for ii in range(len(j[0])):
                total += 1
                if auprc:
                    predictval = softmax(predicts[ii])
                    auprclist.append(
                        (predictval[1].item(), valid_y[ii].item()))
                if classification:
                    if predictlist[ii].index(max(predictlist[ii])) == valid_y[ii]:
                        corrects += 1
        print("test loss: "+str(totalloss/total) +
              ((" acc: "+str(float(corrects)/total)) if classification else ''))
        if auprc:
            print("With AUPRC: "+str(calcAUPRC(auprclist)))
    if classification:
        return float(corrects)/total
    else:
        return (totalloss/total).item()
