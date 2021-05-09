from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn


def train(
    encoder,head,train_dataloader,valid_dataloader,total_epochs,is_packed=False,
    early_stop=False,task="classification",optimtype=torch.optim.RMSprop,lr=0.001,
    weight_decay=0.0,criterion=nn.CrossEntropyLoss(),save_encoder='encoder.pt',
    save_head='head.pt',modalnum=0):

    model = nn.Sequential(encoder,head)
    op = optimtype(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestvalloss = 10000
    bestacc = 0
    bestf1 = 0
    patience = 0

    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out=model([j[0][modalnum].cuda(), j[1][modalnum]])
            else:
                out=model(j[modalnum].float().cuda())
            #print(j[-1])
            loss=criterion(out,j[-1].cuda())
            totalloss += loss * len(j[-1])
            totals+=len(j[-1])
            loss.backward()
            op.step()
        print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            pred = []
            true = []
            for j in valid_dataloader:
                if is_packed:
                    out=model([j[0][modalnum].cuda(), j[1][modalnum]])
                else:
                    out = model(j[modalnum].float().cuda())
                loss = criterion(out,j[-1].cuda())
                totalloss += loss*len(j[-1])
                if task == "classification":
                    pred.append(torch.argmax(out, 1))
                elif task == "multilabel":
                    pred.append(torch.sigmoid(out).round())
                true.append(j[-1])
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        valloss=totalloss/totals
        if task == "classification":
            acc = accuracy_score(true, pred)
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+\
                " acc: "+str(acc))
            if acc>bestacc:
                patience = 0
                bestacc=acc
                print("Saving Best")
                torch.save(encoder,save_encoder)
                torch.save(head,save_head)
            else:
                patience += 1
            if early_stop and patience > 20:
                break
        elif task == "multilabel":
            f1_micro = f1_score(true, pred, average="micro")
            f1_macro = f1_score(true, pred, average="macro")
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+\
                " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
            if f1_macro>bestf1:
                patience = 0
                bestf1=f1_macro
                print("Saving Best")
                torch.save(encoder,save_encoder)
                torch.save(head,save_head)
            else:
                patience += 1
            if early_stop and patience > 20:
                break
        elif task == "regression":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
            if valloss<bestvalloss:
                patience = 0
                bestvalloss=valloss
                print("Saving Best")
                torch.save(encoder,save_encoder)
                torch.save(head,save_head)
            else:
                patience += 1
            if early_stop and patience > 20:
                break
            

def test(encoder,head,test_dataloader,is_packed=False,task="classification",modalnum=0):
    model=nn.Sequential(encoder,head)
    with torch.no_grad():
        totalloss = 0.0
        pred=[]
        true=[]
        for j in test_dataloader:
            if is_packed:
                    out=model([j[0][modalnum].cuda(), j[1][modalnum]])
            else:
                out=model(j[modalnum].float().cuda())
            if task == "classification":
                pred.append(torch.argmax(out, 1))
            elif task == "multilabel":
                pred.append(torch.sigmoid(out).round())
            elif task == "regression":
                loss = nn.L1Loss(out,j[-1].float().cuda())
                totalloss += loss*len(j[-1])
            true.append(j[-1])
        if pred:
            pred = torch.cat(pred, 0).cpu().numpy()
        true = torch.cat(true, 0).cpu().numpy()
        totals = true.shape[0]
        testloss=totalloss/totals
        if task == "classification":
            print("acc: "+str(accuracy_score(true, pred)))
        elif task == "multilabel":
            print(" f1_micro: "+str(f1_score(true, pred, average="micro"))+\
                " f1_macro: "+str(f1_score(true, pred, average="macro")))
        elif task == "regression":
            print("mse: "+str(testloss))
