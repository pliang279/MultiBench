from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from utils.AUPRC import AUPRC
softmax = nn.Softmax()


def train(encoder, head, train_dataloader, valid_dataloader, total_epochs, early_stop=False, optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0, criterion=nn.CrossEntropyLoss(), auprc=False, save_encoder='encoder.pt', save_head='head.pt', modalnum=0, task='classification'):
    model = nn.Sequential(encoder, head)
    op = optimtype(model.parameters(), lr=lr, weight_decay=weight_decay)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0
    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        for j in train_dataloader:
            op.zero_grad()
            out = model(j[modalnum].float().cuda())
            
            if type(criterion) == torch.nn.modules.loss.BCEWithLogitsLoss:
                loss = criterion(out, j[-1].float().cuda())
            else:
                loss = criterion(out, j[-1].cuda())
            totalloss += loss * len(j[-1])
            totals += len(j[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 8)
            op.step()
        print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            pts = []
            for j in valid_dataloader:
                out = model(j[modalnum].float().cuda())
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
        valloss = totalloss/totals
        if task == "classification":
            acc = accuracy_score(true, pred)
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                  " acc: "+str(acc))
            if acc > bestacc:
                patience = 0
                bestacc = acc
                print("Saving Best")
                torch.save(encoder, save_encoder)
                torch.save(head, save_head)
            else:
                patience += 1
        elif task == "multilabel":
            f1_micro = f1_score(true, pred, average="micro")
            f1_macro = f1_score(true, pred, average="macro")
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                  " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
            if f1_macro > bestf1:
                patience = 0
                bestf1 = f1_macro
                print("Saving Best")
                torch.save(encoder, save_encoder)
                torch.save(head, save_head)
            else:
                patience += 1
        elif task == "regression":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
            if valloss < bestvalloss:
                patience = 0
                bestvalloss = valloss
                print("Saving Best")
                torch.save(encoder, save_encoder)
                torch.save(head, save_head)
            else:
                patience += 1
        if early_stop and patience > 7:
            break
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))


def test(encoder, head, test_dataloader, auprc=False, modalnum=0, task='classification', criterion=None):
    model = nn.Sequential(encoder, head)
    with torch.no_grad():
        pred = []
        true = []
        totalloss = 0
        pts = []
        for j in test_dataloader:
            out = model(j[modalnum].float().cuda())
            if criterion is not None:
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
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if criterion is not None:
            print("loss: " + str(totalloss / totals))
        if task == "classification":
            print("acc: "+str(accuracy_score(true, pred)))
            return accuracy_score(true, pred)
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro")) +
                  " f1_macro: "+str(f1_score(true, pred, average="macro")))
            return f1_score(true, pred, average="micro"), f1_score(true, pred, average="macro"), accuracy_score(true, pred)
        else:
            return (totalloss / totals).item()
