import torch
import torchvision
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
r3d = torchvision.models.video.r3d_18(pretrained=True)
model = r3d.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
datas = torch.load('/home/pliang/yiwei/kinetics_small/valid/batch0.pkt')
epochs = 15
valid_dataloader = DataLoader(datas, shuffle=False, batch_size=5)
bestvaloss = 1000
criterion = torch.nn.CrossEntropyLoss()
# a=input()
train_data = []
valid_data = []
test_data = []
with torch.no_grad():
    totalloss = 0.0
    total = 0
    for i in range(24):
        print(" subiter "+str(i))
        datas = torch.load(
            '/home/pliang/yiwei/kinetics_small/train/batch'+str(i)+'.pkt')
        train_dataloader = DataLoader(datas, shuffle=True, batch_size=5)
        for j in train_dataloader:
            out = model(j[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            
            for ii in range(len(j[0])):
                train_data.append([out[ii].cpu(), j[1][ii]])
    with torch.no_grad():
        total = 0
        correct = 0
        totalloss = 0.0
        for j in valid_dataloader:
            out = model(j[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
            for ii in range(len(j[0])):
                valid_data.append([out[ii].cpu(), j[1][ii]])

valid_dataloader = None
datas = torch.load('/home/pliang/yiwei/kinetics_small/test/batch0.pkt')
test_dataloader = DataLoader(datas, shuffle=False, batch_size=5)
with torch.no_grad():
    total = 0
    correct = 0
    totalloss = 0.0
    for j in test_dataloader:
        out = model(j[0].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        for ii in range(len(out)):
            test_data.append([out[ii].cpu(), j[1][ii]])

torch.save([train_data, valid_data, test_data],
           '/home/pliang/yiwei/kinetics_small/features.pt')
