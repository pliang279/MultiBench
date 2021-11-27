# NOTE dataloader needs to be implemented
#   Please use special/kinetics_audio_unimodal.py for now
import os
import sys
import torch
import torchvision

sys.path.append(os.getcwd())

from unimodals.common_models import MLP
from datasets.kinetics.get_data import get_dataloader
from training_structures.unimodal import train, test


modalnum = 1
traindata, validdata, testdata = get_dataloader(sys.argv[1])
r50 = torchvision.models.resnet50(pretrained=True)
r50.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder = torch.nn.Sequential(r50, MLP(1000, 200, 64)).cuda()
head = MLP(64, 200, 5).cuda()

train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD,
      lr=0.01, weight_decay=0.0001, modalnum=modalnum)

print("Testing:")
encoder = torch.load('encoder.pt').cuda()
head = torch.load('head.pt')
test(encoder, head, testdata, modalnum=modalnum)
