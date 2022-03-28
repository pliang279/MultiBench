from datasets.avmnist.get_data import get_dataloader
import torch.autograd as A
import torch.nn.functional as F
import torch.nn as nn
import torch
from unimodals.common_models import GlobalPooling2D
import sys
import os
sys.path.append(os.getcwd())


# %%

class GP_LeNet(nn.Module):
    def __init__(self, args, in_channels):
        super(GP_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, args.channels,
                               kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(args.channels))
        self.gp1 = GlobalPooling2D()

        self.conv2 = nn.Conv2d(
            args.channels, 2 * args.channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(int(2 * args.channels))
        self.gp2 = GlobalPooling2D()

        self.conv3 = nn.Conv2d(
            2 * args.channels, 4 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(4 * args.channels))
        self.gp3 = GlobalPooling2D()

        self.classifier = nn.Sequential(
            nn.Linear(int(4 * args.channels), args.num_outputs)
        )

        # initialization of weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out1, 2)
        gp1 = self.gp1(out1)

        out2 = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out2, 2)
        gp2 = self.gp2(out2)

        out3 = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out3, 2)
        gp3 = self.gp3(out3)

        out = self.classifier(gp3)

        return out, gp1, gp2, gp3


class GP_LeNet_Deeper(nn.Module):
    def __init__(self, args, in_channels):
        super(GP_LeNet_Deeper, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, args.channels,
                               kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(int(args.channels))
        self.gp1 = GlobalPooling2D()

        self.conv2 = nn.Conv2d(
            args.channels, 2 * args.channels, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(int(2 * args.channels))
        self.gp2 = GlobalPooling2D()

        self.conv3 = nn.Conv2d(
            2 * args.channels, 4 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(4 * args.channels))
        self.gp3 = GlobalPooling2D()

        self.conv4 = nn.Conv2d(
            4 * args.channels, 8 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(8 * args.channels))
        self.gp4 = GlobalPooling2D()

        self.conv5 = nn.Conv2d(
            8 * args.channels, 16 * args.channels, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(16 * args.channels))
        self.gp5 = GlobalPooling2D()

        self.classifier = nn.Sequential(
            nn.Linear(int(16 * args.channels), args.num_outputs)
        )

        # initialization of weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out1, 2)
        gp1 = self.gp1(out)

        out2 = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out2, 2)
        gp2 = self.gp2(out2)

        out3 = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out3, 2)
        gp3 = self.gp3(out3)

        out4 = F.relu(self.bn4(self.conv4(out)))
        out = F.max_pool2d(out4, 2)
        gp4 = self.gp4(out4)

        out5 = F.relu(self.bn5(self.conv5(out)))
        out = F.max_pool2d(out5, 2)
        gp5 = self.gp5(out5)

        out = self.classifier(gp5)

        return out, gp1, gp2, gp3, gp4, gp5


class SimpleAVNet(nn.Module):
    def __init__(self, args, audio_channels, image_channels):
        super(SimpleAVNet, self).__init__()

        self.audio_net = GP_LeNet(args, audio_channels)
        self.image_net = GP_LeNet(args, image_channels)

        self.classifier = nn.Linear(
            int(2 * 4 * args.channels), args.num_outputs)

    def forward(self, audio, image):
        audio_out, audio_gp1, audio_gp2, audio_gp3 = self.audio_net(audio)
        image_out, image_gp1, image_gp2, image_gp3 = self.image_net(image)

        multimodal_feat = torch.cat((audio_gp3, image_gp3), 1)
        out = self.classifier(multimodal_feat)

        return out


class SimpleAVNet_Deeper(nn.Module):
    def __init__(self, args, audio_channels, image_channels):
        super(SimpleAVNet_Deeper, self).__init__()

        self.audio_net = GP_LeNet_Deeper(args, audio_channels)
        self.image_net = GP_LeNet(args, image_channels)

        self.classifier = nn.Linear(int(20 * args.channels), args.num_outputs)

    def forward(self, audio, image):
        audio_out, audio_gp1, audio_gp2, audio_gp3, audio_gp4, audio_gp5 = self.audio_net(
            audio)
        image_out, image_gp1, image_gp2, image_gp3 = self.image_net(image)

        multimodal_feat = torch.cat((audio_gp5, image_gp3), 1)
        out = self.classifier(multimodal_feat)

        return out


class Help:
    def __init__(self):
        self.channels = 3
        self.num_outputs = 10


model = SimpleAVNet_Deeper(Help(), 1, 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
optim = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
trains, valids, tests = get_dataloader('/data/yiwei/avmnist/_MFAS/avmnist')
criterion = nn.CrossEntropyLoss()
for ep in range(100):
    totalloss = 0.0
    batches = 0
    for j in trains:
        batches += 1
        optim.zero_grad()
        inputs = [x.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for x in j[:-1]]
        labels = j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        preds = model(inputs[1], inputs[0])
        loss = criterion(preds, labels)
        loss.backward()
        optim.step()
        totalloss += loss
    print("ep "+str(ep) + " train loss "+str(totalloss/batches))
    batches = 0
    total = 0
    corrects = 0
    totalloss = 0
    with torch.no_grad():
        for j in valids:
            batches += 1
            inputs = [x.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for x in j[:-1]]
            labels = j[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            preds = model(inputs[1], inputs[0])
            loss = criterion(preds, labels)
            totalloss += loss
            for i in range(len(j[-1])):
                total += 1
                if torch.argmax(preds[i]).item() == j[-1][i].item():
                    corrects += 1
    print("ep "+str(ep)+" valid loss "+str(totalloss/batches) +
          " acc: "+str(float(corrects)/total))
