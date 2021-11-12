from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from utils.AUPRC import AUPRC
from objective_functions.cca import CCALoss
#import pdb

softmax = nn.Softmax()


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
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i], training=training))
        out = self.fuse(outs, training=training)
        
        return outs, self.head(out, training=training)


def train(
        encoders, fusion, head, train_dataloader, valid_dataloader, total_epochs, is_packed=False, outdim=10,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        criterion=nn.CrossEntropyLoss(), auprc=False, save='best.pt'):

    model = MMDL(encoders, fusion, head, is_packed).cuda()
    op = optimtype([p for p in model.parameters()
                   if p.requires_grad], lr=lr, weight_decay=weight_decay)
    #scheduler = ExponentialLR(op, 0.9)
    cca_criterion = CCALoss(outdim, False, device=torch.device("cuda"))

    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0

    for epoch in range(total_epochs):
        totalloss = 0.0
        totalloss1 = 0.0
        totalloss2 = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out1, out2 = model(
                        [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()], training=True)
                    
                    loss1 = criterion(out2, j[-1].cuda())
                    loss2 = cca_criterion(out1[0], out1[1])
                    loss = loss1+1e-3*loss2
            else:
                out1, out2 = model([i.float().cuda()
                                   for i in j[:-1]], training=True)
                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss1 = criterion(out2, j[-1].float().cuda())
                else:
                    if len(j[-1].size()) > 1:
                        j[-1] = j[-1].squeeze()
                    loss1 = criterion(out2, j[-1].long().cuda())
                loss2 = cca_criterion(out1[0], out1[1])
                loss = loss1+1e-3*loss2
            
            totalloss += loss * len(j[-1])
            totals += len(j[-1])
            totalloss1 += loss1 * len(j[-1])
            totalloss2 += loss2 * len(j[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            op.step()
        print("Epoch "+str(epoch)+" train loss: " +
              str(totalloss1/totals)+" cca loss: "+str(totalloss2/totals))

        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            pts = []
            for j in valid_dataloader:
                if is_packed:
                    _, out = model(
                        [[i.cuda() for i in j[0]], j[1]], training=False)
                else:
                    _, out = model([i.float().cuda()
                                   for i in j[:-1]], training=False)
                if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                    loss = criterion(out, j[-1].float().cuda())
                else:
                    if len(j[-1].size()) > 1:
                        j[-1] = j[-1].squeeze()
                    loss = criterion(out, j[-1].long().cuda())
                totalloss += loss*len(j[-1])
                
                if task == "classification":
                    pred.append(torch.argmax(out, 1))
                elif task == "multilabel":
                    pred.append(torch.sigmoid(out).round())
                true.append(j[-1])
                if auprc:
                    # pdb.set_trace()
                    sm = softmax(out)
                    pts += [(sm[i][1].item(), j[-1][i].item())
                            for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        valloss = totalloss/totals
        if task == "classification":
            acc = accuracy_score(true, pred)
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                  " acc: "+str(acc))
            if valloss < bestvalloss:
                patience = 0
                bestvalloss = valloss
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
            if early_stop and patience > 7:
                break
        elif task == "multilabel":
            f1_micro = f1_score(true, pred, average="micro")
            f1_macro = f1_score(true, pred, average="macro")
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                  " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
            if f1_macro > bestf1:
                patience = 0
                bestf1 = f1_macro
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
            if early_stop and patience > 7:
                break
        elif task == "regression":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
            if valloss < bestvalloss:
                patience = 0
                bestvalloss = valloss
                print("Saving Best")
                torch.save(model, save)
            else:
                patience += 1
            if early_stop and patience > 7:
                break
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))

        # scheduler.step()


def test(
        model, test_dataloader, is_packed=False,
        criterion=nn.CrossEntropyLoss(), task="classification", auprc=False):
    with torch.no_grad():
        totalloss = 0.0
        pred = []
        true = []
        pts = []
        for j in test_dataloader:
            if is_packed:
                _, out = model(
                    [[i.cuda() for i in j[0]], j[1]], training=False)
            else:
                _, out = model([i.float().cuda()
                               for i in j[:-1]], training=False)
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                loss = criterion(out, j[-1].float().cuda())
            else:
                loss = criterion(out, j[-1].cuda())
            
            totalloss += loss*len(j[-1])
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            true.append(j[-1])
            if auprc:
                # pdb.set_trace()
                sm = softmax(out)
                pts += [(sm[i][1].item(), j[-1][i].item())
                        for i in range(j[-1].size(0))]
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        testloss = totalloss/totals
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if task == "classification":
            print("acc: "+str(accuracy_score(true, pred)))
            return accuracy_score(true, pred)
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return f1_score(true, pred, average="micro"), f1_score(true, pred, average="macro"), accuracy_score(true, pred)
        elif task == "regression":
            print("mse: "+str(testloss))
            return testloss
