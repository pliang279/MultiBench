import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.AUPRC import AUPRC
from objective_functions.regularization import RegularizationLoss
#import pdb

softmax = nn.Softmax()

class MMDL(nn.Module):
    def __init__(self,encoders,fusion,head,has_padding=False):
        super(MMDL,self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding=has_padding
    
    def forward(self,inputs,training=False):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i],inputs[1][i]], training=training))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i], training=training))
        out = self.fuse(outs, training=training)
        return self.head(out, training=training)


def train(
    encoders,fusion,head,train_dataloader,valid_dataloader,total_epochs,
    task="classification",optimtype=torch.optim.RMSprop,lr=0.001,weight_decay=0.0,
    criterion=nn.CrossEntropyLoss(),regularization=False,auprc=False,save='best.pt'):
    
    model = MMDL(encoders,fusion,head,regularization).cuda()
    op = optimtype(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestvalloss = 10000
    
    if regularization:
        regularize = RegularizationLoss(criterion, model)
    
    for epoch in range(total_epochs):
        totalloss = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            #print([i for i in j[:-1]])
            op.zero_grad()
            if regularization:
                with torch.backends.cudnn.flags(enabled=False):
                    out=model([[i.cuda() for i in j[0]], j[1]],training=True)
                    #print(j[-1])
                    loss=criterion(out,j[-1].cuda()) + regularize(out, [[i.cuda() for i in j[0]], j[1]])
            else:
                out=model([i.float().cuda() for i in j[:-1]],training=True)
                #print(j[-1])
                loss=criterion(out,j[-1].cuda())
            totalloss += loss * len(j[-1])
            totals+=len(j[-1])
            if regularization:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            op.step()
        print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            totals = 0
            correct = 0
            pts = []
            for j in valid_dataloader:
                if regularization:
                    out=model([[i.cuda() for i in j[0]], j[1]],training=False)
                else:
                    out = model([i.float().cuda() for i in j[:-1]],training=False)
                loss = criterion(out,j[-1].cuda())
                totalloss += loss*len(j[-1])
                for i in range(len(j[-1])):
                    totals += 1
                    if task == "classification":
                        if torch.argmax(out[i]).item() == j[-1][i].item():
                            correct += 1
                    if auprc:
                        #pdb.set_trace()
                        sm=softmax(out[i])
                        pts.append((sm[1].item(), j[-1][i].item()))
        valloss=totalloss/totals
        if task == "classification":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss)+" acc: "+str(float(correct)/totals))
        elif task == "regression":
            print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if valloss<bestvalloss:
            bestvalloss=valloss
            print("Saving Best")
            torch.save(model,save)

def test(
    model,test_dataloader,criterion=nn.CrossEntropyLoss(),
    task="classification",regularization=False,auprc=False):
    with torch.no_grad():
        totals=0
        correct=0
        totalloss = 0.0
        pts=[]
        for j in test_dataloader:
            if regularization:
                out=model([[i.cuda() for i in j[0]], j[1]],training=False)
            else:
                out = model([i.float().cuda() for i in j[:-1]],training=False)
            loss = criterion(out,j[-1].cuda())
            totalloss += loss*len(j[-1])
            for i in range(len(j[-1])):
                totals += 1
                if task == "classification":
                    if torch.argmax(out[i]).item()==j[-1][i].item():
                        correct += 1
                if auprc:
                    sm=softmax(out[i])
                    pts.append((sm[1].item(),j[-1][i].item()))
        testloss=totalloss/totals
        if task == "classification":
            print("acc: "+str(float(correct)/totals))
        elif task == "regression":
            print("mse: "+str(testloss))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))

