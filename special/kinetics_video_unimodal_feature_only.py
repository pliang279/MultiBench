from unimodals.common_models import MLP
import torch
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
model = MLP(400, 200, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
optim = torch.optim.Adam(model.parameters(), lr=0.01)
train_datas, valid_datas, test_datas = torch.load(
    '/home/pliang/yiwei/features/features.pt')
epochs = 30
valid_dataloader = DataLoader(valid_datas, shuffle=False, batch_size=40)
test_dataloader = DataLoader(test_datas, shuffle=False, batch_size=40)
train_dataloader = DataLoader(train_datas, shuffle=True, batch_size=40)

bestvaloss = 1000
criterion = torch.nn.CrossEntropyLoss()
# a=input()
for ep in range(epochs):
    totalloss = 0.0
    total = 0
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
            totalloss += loss * len(j[0])
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
            torch.save(model, 'best1.pt')

print('testing')
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
