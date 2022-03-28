# NOTE dataloader needs to be implemented
#   Please use special/kinetics_video_unimodal.py for now
import os
import sys
import torch

sys.path.append(os.getcwd())

from unimodals.common_models import ResNetLSTMEnc, MLP
from datasets.kinetics.get_data import get_dataloader
from training_structures.unimodal import train, test



modalnum = 0
traindata, validdata, testdata = get_dataloader(sys.argv[1])
encoder = ResNetLSTMEnc(64).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = MLP(64, 200, 5).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD,
      lr=0.01, weight_decay=0.0001, modalnum=modalnum)

print("Testing:")
encoder = torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = torch.load('head.pt')
test(encoder, head, testdata, modalnum=modalnum)
