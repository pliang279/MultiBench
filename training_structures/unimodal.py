import torch
from torch import nn
from utils.AUPRC import AUPRC
softmax = nn.Softmax()

def train(encoder,head,train_dataloader,valid_dataloader,total_epochs,optimtype=torch.optim.RMSprop,lr=0.001,weight_decay=0.0,criterion=nn.CrossEntropyLoss(),auprc=False,save_encoder='encoder.pt',save_head='head.pt',modalnum=0,task='classification'):
    model = nn.Sequential(encoder,head)
    op = optimtype(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestvalloss = 10000
    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        for j in train_dataloader:
            op.zero_grad()
            out=model(j[modalnum].float().cuda())
            #print(j[-1])
            loss=criterion(out,j[-1].cuda())
            totalloss += loss * len(j[-1])
            totals+=len(j[-1])
            loss.backward()
            op.step()
        print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        with torch.no_grad():
            totalloss = 0.0
            totals = 0
            correct = 0
            pts = []
            for j in valid_dataloader:
                out = model(j[modalnum].float().cuda())
                loss = criterion(out,j[-1].cuda())
                totalloss += loss*len(j[-1])
                for i in range(len(j[-1])):
                    totals += 1
                    if task == 'classification':
                        if torch.argmax(out[i]).item() == j[-1][i].item():
                            correct += 1
                        if auprc:
                            #pdb.set_trace()
                            sm=softmax(out[i])
                            pts.append((sm[1].item(), j[-1][i].item()))
        valloss=totalloss/totals
        if task == 'classification':
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+" acc: "+str(float(correct)/totals))
        else:
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if valloss<bestvalloss:
            bestvalloss=valloss
            print("saving best")
            torch.save(encoder,save_encoder)
            torch.save(head,save_head)

def test(encoder,head,test_dataloader,auprc=False,modalnum=0,task='classification',criterion=None):
    model=nn.Sequential(encoder,head)
    with torch.no_grad():
        totals=0
        correct=0
        totalloss=0
        pts=[]
        for j in test_dataloader:
            out=model(j[modalnum].float().cuda())
            if criterion is not None:
                loss = criterion(out,j[-1].cuda())
                totalloss += loss*len(j[-1])
            for i in range(len(j[-1])):
                totals += 1
                if task == 'classification':
                    if torch.argmax(out[i]).item()==j[-1][i].item():
                        correct += 1
                    if auprc:
                        sm=softmax(out[i])
                        pts.append((sm[1].item(),j[-1][i].item()))
        if criterion is not None:
            print("loss: " + str(totalloss / totals))
        print("acc: "+str(float(correct)/totals))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
