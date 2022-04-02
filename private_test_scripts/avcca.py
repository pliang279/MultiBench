from unimodals.common_models import LeNet, MLP, Constant
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from utils.helper_modules import Sequential2
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from training_structures.cca_onestage import train, test
from torch import nn
from datasets.avmnist.get_data import get_dataloader
import torch
import sys
import os
sys.path.append(os.getcwd())

traindata, validdata, testdata = get_dataloader(
    '/home/pliang/yiwei/avmnist/_MFAS/avmnist', batch_size=800)
channels = 6
encoders = [LeNet(1, channels, 3).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), Sequential2(
    LeNet(1, channels, 5), Linear(192, 48, xavier_init=True)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
#encoders=[MLP(300,512,outdim), MLP(4096,1024,outdim)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
# head=MLP(2*outdim,2*outdim,23).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = Linear(96, 10, xavier_init=True).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def trpr():
    train(encoders, fusion, head, traindata, validdata, 25, outdim=48,
          save="best_cca.pt", optimtype=torch.optim.AdamW, lr=1e-2)

    # ,weight_decay=0.01)
all_in_one_train(trpr, encoders+[fusion, head])
print("Testing:")
model = torch.load('best_cca.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def tepr():
    test(model, testdata)


all_in_one_test(tepr, [model])
