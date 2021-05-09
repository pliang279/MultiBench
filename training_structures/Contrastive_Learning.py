from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from utils.AUPRC import AUPRC
from objective_functions.contrast import NCEAverage, NCECriterion
#import pdb


softmax = nn.Softmax()


class MMDL(nn.Module):
    def __init__(self,encoders,fusion,head,n_data,has_padding=False):
        super(MMDL,self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.has_padding=has_padding
        self.contrast = NCEAverage(200, n_data, 16384)
    
    def forward(self,inputs,classifier=False,training=False):
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i]([inputs[0][i],inputs[1][i]], training=training))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i], training=training))
        
        if classifier:
            #out = self.fuse(outs, training=training)
            return self.head(outs[1], training=training)
        else:
            out1, out2 = self.contrast(outs[0], outs[1], inputs[2])
            return out1, out2 
        '''

        out = self.fuse(outs, training=training)
        out1, out2 = self.contrast(outs[0], outs[1], inputs[2])

        return self.head(out, training=training), out1, out2    
        '''

def train(
    encoders,fusion,head,train_dataloader,valid_dataloader,total_epochs,is_packed=False,
    early_stop=False,optimtype=torch.optim.RMSprop,lr=0.001,weight_decay=0.0,
    criterion=nn.L1Loss(),auprc=False,save='best.pt'):
    
    n_data = len(train_dataloader.dataset)
    model = MMDL(encoders,fusion,head,n_data,is_packed).cuda()
    op = optimtype(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler = ExponentialLR(op, 0.9)
    contrast_criterion = NCECriterion(n_data)

    bestloss = 0.0
    patience = 0
    
    for epoch in range(10000):
        totalloss = 0.0
        totals = 0
        model.train()
        
        #Representation Learning
        for j in train_dataloader:
            #print([i for i in j[:-1]])
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out1, out2=model(
                        [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],training=True)
                    #print(j[-1])
                    loss1=contrast_criterion(out1)
                    loss2=contrast_criterion(out2)
                    loss = loss1+loss2
            else:
                out=model([i.float().cuda() for i in j[:-1]],training=True)
                #print(j[-1])
                loss=criterion(out,j[-1].cuda())
            totalloss += loss * len(j[-1])
            totals+=len(j[-1])
            
            loss.backward()
            op.step()
        
        train_loss = totalloss/totals
        print("Epoch "+str(epoch)+" train loss: "+str(train_loss))

        if bestloss == 0.0:
            bestloss = train_loss
            continue

        if (bestloss-train_loss)/bestloss < 1e-6:
            patience += 1
            print(patience)
        else:
            bestloss = train_loss
            patience = 0
        if patience > 20:
            print("Early Stop!")
            break
    
    patience = 0
    bestvalloss = 10000
    for epoch in range(total_epochs):
        totalloss = 0.0
        total_reg_loss = 0.0
        total_contrast_loss = 0.0
        totals = 0
        model.train()
        for j in train_dataloader:
            #print([i for i in j[:-1]])
            op.zero_grad()
            if is_packed:
                with torch.backends.cudnn.flags(enabled=False):
                    out=model(
                        [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],True,training=True)
                    #out, out1, out2=model(
                    #    [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],True,training=True)
                    #print(j[-1])
            else:
                out=model([i.float().cuda() for i in j[:-1]],training=True)
                #print(j[-1])
            reg_loss=criterion(out,j[-1].cuda())
            #loss1=contrast_criterion(out1)
            #loss2=contrast_criterion(out2)
            #contrast_loss = loss1 + loss2
            #loss = reg_loss+0.1*contrast_loss
            loss = reg_loss
            total_reg_loss += reg_loss * len(j[-1])
            #total_contrast_loss += contrast_loss * len(j[-1])
            totalloss += loss * len(j[-1])
            totals+=len(j[-1])
            
            loss.backward()
            op.step()
        #print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
        print("Epoch "+str(epoch)+" train loss: "+str(total_reg_loss/totals)+" contrast loss: "+str(total_contrast_loss/totals))
        
        model.eval()
        with torch.no_grad():
            totalloss = 0.0
            totals = 0
            pts = []
            for j in valid_dataloader:
                if is_packed:
                    out=model(
                        [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],True,training=False)
                    #out, _, _=model(
                    #    [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],True,training=False)
                else:
                    out = model([i.float().cuda() for i in j[:-1]],training=False)
                loss = criterion(out,j[-1].cuda())
                totalloss += loss*len(j[-1])
                for i in range(len(j[-1])):
                    totals += 1
                    if auprc:
                        #pdb.set_trace()
                        sm=softmax(out[i])
                        pts.append((sm[1].item(), j[-1][i].item()))
        valloss=totalloss/totals
        print("Epoch "+str(epoch)+" valid loss: "+str(valloss))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
        if valloss<bestvalloss:
            patience = 0
            bestvalloss=valloss
            print("Saving Best")
            torch.save(model,save)
        else:
            patience += 1
        if early_stop and patience > 20:
            break
        
        scheduler.step()


def test(
    model,test_dataloader,is_packed=False,
    criterion=nn.L1Loss(),auprc=False):
    with torch.no_grad():
        totals=0
        totalloss = 0.0
        pts=[]
        for j in test_dataloader:
            if is_packed:
                out=model(
                    [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],True,training=False)
                #out, _, _=model(
                #    [[j[0][0].cuda(), j[0][2].cuda()], j[1], j[2].cuda()],True,training=False)
            else:
                out = model([i.float().cuda() for i in j[:-1]],training=False)
            loss = criterion(out,j[-1].cuda())
            #print(torch.cat([out,j[-1].cuda()],dim=1))
            totalloss += loss*len(j[-1])
            for i in range(len(j[-1])):
                totals += 1
                if auprc:
                    sm=softmax(out[i])
                    pts.append((sm[1].item(),j[-1][i].item()))
        testloss=totalloss/totals
        print("mse: "+str(testloss))
        if auprc:
            print("AUPRC: "+str(AUPRC(pts)))
