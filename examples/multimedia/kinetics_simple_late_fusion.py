# NOTE dataloader needs to be implemented
#   Please use special/kinetics_simple_late_fusion.py for now
import sys
import os
import torch
import torchvision

sys.path.append(os.getcwd())

from unimodals.common_models import ResNetLSTMEnc, MLP
from datasets.kinetics.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test



traindata, validdata, testdata = get_dataloader(sys.argv[1])
r50 = torchvision.models.resnet50(pretrained=True)
r50.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
audio_encoder = torch.nn.Sequential(r50, MLP(1000, 200, 64)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
encoders = [ResNetLSTMEnc(64).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), audio_encoder.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
head = MLP(64+64, 200, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

train(encoders, fusion, head, traindata, validdata, 30,
      optimtype=torch.optim.SGD, lr=0.1, weight_decay=0.0001)

print("Testing:")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
test(model, testdata)
