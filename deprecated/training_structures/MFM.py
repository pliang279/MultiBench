from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn
from utils.AUPRC import AUPRC


class MFM(nn.Module):
    def __init__(self, encoders, decoders, fusionmodule, head, intermediates):
        super(MFM, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.intermediates = nn.ModuleList(intermediates)
        self.fuse = fusionmodule
        self.head = head

    def forward(self, inputs):
        outs = []
        for i in range(len(inputs)):
            outs.append(self.encoders[i](inputs[i]))
        
        
        fused = self.fuse(outs)
        
        combined = self.intermediates[-1](fused)
        recons = []
        for i in range(len(outs)):
            outs[i] = self.intermediates[i](outs[i])
        for i in range(len(inputs)):
            recons.append(self.decoders[i](
                torch.cat([outs[i], combined], dim=1)))
        return recons, self.head(combined)


def train_MFM(
        encoders, decoders, head, intermediates, fusion, recon_loss_func, train_dataloader, valid_dataloader,
        total_epochs, ce_weight=2.0, learning_rate=0.001, savedir='best.pt',
        early_stop=False, task="classification", criterion=nn.CrossEntropyLoss()):

    n_modals = len(encoders)
    bestvalloss = 100
    bestf1 = 0
    patience = 0
    mvae = MFM(encoders, decoders, fusion, head, intermediates)
    optim = torch.optim.Adam(mvae.parameters(), lr=learning_rate)
    for ep in range(total_epochs):
        totalloss = 0.0
        total = 0
        for j in train_dataloader:
            optim.zero_grad()
            trains = [x.float().cuda() for x in j[:-1]]
            total += len(trains[0])
            mvae.train()
            recons, outs = mvae(trains)
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                loss = criterion(outs, j[-1].float().cuda())*ce_weight
            else:
                loss = criterion(outs, j[-1].cuda())*ce_weight
            
            loss += recon_loss_func(recons, trains)
            
            loss.backward()
            totalloss += loss*len(trains[0])
            optim.step()
        print("epoch "+str(ep)+" train loss: "+str(totalloss/total))
        if True:
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                for j in valid_dataloader:
                    trains = [x.float().cuda() for x in j[:-1]]
                    mvae.train()
                    _, outs = mvae(trains)
                    if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                        loss = criterion(outs, j[-1].float().cuda())
                    else:
                        loss = criterion(outs, j[-1].cuda())
                    totalloss += loss * len(j[0])
                    if task == "classification":
                        pred.append(torch.argmax(outs, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(outs).round())
                    true.append(j[-1])
                if pred:
                    pred = torch.cat(pred, 0).cpu().numpy()
                true = torch.cat(true, 0).cpu().numpy()
                total = true.shape[0]
                valoss = totalloss/total
                if task == "classification":
                    acc = accuracy_score(true, pred)
                    print("Epoch "+str(ep)+" valid loss: "+str(valoss) +
                          " acc: "+str(acc))
                    if valoss < bestvalloss:
                        patience = 0
                        bestvalloss = valoss
                        print("Saving Best")
                        torch.save(mvae, savedir)
                    else:
                        patience += 1
                elif task == "multilabel":
                    f1_micro = f1_score(true, pred, average="micro")
                    f1_macro = f1_score(true, pred, average="macro")
                    print("Epoch "+str(ep)+" valid loss: "+str(valoss) +
                          " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                    if f1_macro > bestf1:
                        patience = 0
                        bestf1 = f1_macro
                        print("Saving Best")
                        torch.save(mvae, savedir)
                    else:
                        patience += 1
                if early_stop and patience > 7:
                    break


def test_MFM(model, test_dataloader, auprc=False, task="classification"):
    pred = []
    true = []
    pts = []
    for j in test_dataloader:
        xes = [x.float().cuda() for x in j[:-1]]
        y_batch = j[-1].cuda()
        with torch.no_grad():
            model.eval()
            _, outs = model(xes)
        if task == "classification":
            a = nn.Softmax()(outs)
            for ii in range(len(outs)):
                pts.append([a[ii][1], y_batch[ii]])
            pred.append(torch.argmax(outs, 1))
        elif task == "multilabel":
            pred.append(torch.sigmoid(outs).round())
        true.append(j[-1])
    if pred:
        pred = torch.cat(pred, 0).cpu().numpy()
    true = torch.cat(true, 0).cpu().numpy()
    if auprc:
        print(AUPRC(pts))
    if task == "classification":
        print("acc: "+str(accuracy_score(true, pred)))
        return accuracy_score(true, pred)
    elif task == "multilabel":
        print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
              " f1_macro: "+str(f1_score(true, pred, average="macro")))
        return f1_score(true, pred, average="micro"), f1_score(true, pred, average="macro")
