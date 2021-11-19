from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data_robust import get_dataloader
from training_structures.unimodal import train, test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

filename_encoder = 'avmnist_unimodal_0_encoder.pt'
filename_head = 'avmnist_unimodal_0_head.pt'
modalnum = 0
traindata, validdata, testdata, robustdata = get_dataloader(
    '../../../../yiwei/avmnist/_MFAS/avmnist')
channels = 3
# encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
encoder = LeNet(1, channels, 3).cuda()
head = MLP(channels*8, 100, 10).cuda()


train(encoder, head, traindata, validdata, 20, optimtype=torch.optim.SGD, lr=0.01,
      weight_decay=0.0001, modalnum=modalnum, save_encoder=filename_encoder, save_head=filename_head)

encoder = torch.load(filename_encoder).cuda()
head = torch.load(filename_head)
print("Testing:")
test(encoder, head, testdata, modalnum=modalnum)

print("Robustness testing:")
test(encoder, head, robustdata, modalnum=modalnum)
