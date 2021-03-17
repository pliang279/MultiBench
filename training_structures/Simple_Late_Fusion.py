import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.AUPRC import AUPRC
import pdb

softmax = nn.Softmax()

class MMDL(nn.Module):
    def __init__(self,encoders,fusion,head):
        super(MMDL,self).__init__()
        self.encoders=nn.ModuleList(encoders)
        self.fuse=fusion
        self.head=head
    
    def forward(self,inputs,training=False):
        outs=[]
        for i in range(len(inputs)):
            outs.append(self.encoders[i](inputs[i],training=training))
        out=self.fuse(outs,training=training)
        return self.head(out,training=training)

def train(encoders,fusion,head,train_dataloader,valid_dataloader,total_epochs,optimtype=torch.optim.RMSprop,lr=0.001,weight_decay=0.0,criterion=nn.CrossEntropyLoss(),auprc=False,save='best.pt'):
    model = MMDL(encoders,fusion,head).cuda()
    op = optimtype(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestvalloss = 10000
    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        for j in train_dataloader:
            op.zero_grad()
            out=model([i.float().cuda() for i in j[:-1]])
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
                out = model([i.float().cuda() for i in j[:-1]])
                loss = criterion(out,j[-1].cuda())
                totalloss += loss*len(j[-1])
                for i in range(len(j[-1])):
                    totals += 1
                    if torch.argmax(out[i]).item() == j[-1][i].item():
                        correct += 1
                    if auprc:
                        pdb.set_trace()
                        sm=softmax(out[i])
                        pts.append((sm[1].item(), j[-1][i].item()))
        valloss=totalloss/totals
        print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+" acc: "+str(float(correct)/totals))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if valloss<bestvalloss:
            bestvalloss=valloss
            print("Saving Best")
            torch.save(model,save)

def test(model,test_dataloader,auprc=False):
    with torch.no_grad():
        totals=0
        correct=0
        pts=[]
        for j in test_dataloader:
            out=model([i.float().cuda() for i in j[:-1]])
            for i in range(len(j[-1])):
                totals += 1
                if torch.argmax(out[i]).item()==j[-1][i].item():
                    correct += 1
                if auprc:
                    sm=softmax(out[i])
                    pts.append((sm[1].item(),j[-1][i].item()))
        print("acc: "+str(float(correct)/totals))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))

