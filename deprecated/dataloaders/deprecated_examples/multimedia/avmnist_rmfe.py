from unimodals.common_models import LeNet, MLP, Constant
from utils.helper_modules import Sequential2
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
from torch import nn
from datasets.avmnist.get_data import get_dataloader
import torch
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist', batch_size=2)
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
#encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
# head=MLP(2*outdim,2*outdim,23).cuda()
head = MLP(channels*40, 100, 10).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 15, regularization=True,
      save="best_reg.pt", optimtype=torch.optim.AdamW, lr=0.01, weight_decay=0.01)

print("Testing:")
model = torch.load('best_cca.pt').cuda()
test(model, testdata)
