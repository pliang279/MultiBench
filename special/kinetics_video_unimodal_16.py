import torch
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
#r3d = torchvision.models.video.r3d_18(pretrained=True)
# model=torch.nn.Sequential(r3d,torch.load('best1.pt')).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
model = torch.load('best2.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch0.pdt')
criterion = torch.nn.CrossEntropyLoss()

epochs = 15
valid_dataloader = DataLoader(datas, shuffle=False, batch_size=45)
bestvaloss = 1000
# a=input()
for ep in range(epochs):
    totalloss = 0.0
    total = 0
    for i in range(24):
        print("epoch "+str(ep)+" subiter "+str(i))
        datas = torch.load(
            '/home/pliang/yiwei/kinetics_small/train/batch'+str(i)+'.pdt')
        train_dataloader = DataLoader(datas, shuffle=True, batch_size=45)
        for j in train_dataloader:
            optim.zero_grad()
            out = model(j[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            
            loss = criterion(out, j[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            loss.backward()
            optim.step()
            totalloss += loss*len(j[0])
            total += len(j[0])
    print("Epoch "+str(ep)+" train loss: "+str(totalloss/total))
    with torch.no_grad():
        total = 0
        correct = 0
        totalloss = 0.0
        for j in valid_dataloader:
            out = model(j[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            loss = criterion(out, j[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            totalloss += loss
            for ii in range(len(out)):
                total += 1
                if out[ii].tolist().index(max(out[ii])) == j[1][ii]:
                    correct += 1
        valoss = totalloss/total
        print("Valid loss: "+str(totalloss/total) +
              " acc: "+str(float(correct)/total))
        if valoss < bestvaloss:
            print("Saving best")
            bestvaloss = valoss
            torch.save(model, 'best16.pt')

print('testing')
valid_dataloader = None
datas = torch.load('/home/pliang/yiwei/kinetics_small/test/batch0.pdt')
test_dataloader = DataLoader(datas, shuffle=False, batch_size=45)
with torch.no_grad():
    total = 0
    correct = 0
    totalloss = 0.0
    for j in test_dataloader:
        out = model(j[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        loss = criterion(out, j[1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        totalloss += loss
        for ii in range(len(out)):
            total += 1
            if out[ii].tolist().index(max(out[ii])) == j[1][ii]:
                correct += 1
    print("Test loss: "+str(totalloss/total) +
          " acc: "+str(float(correct)/total))
