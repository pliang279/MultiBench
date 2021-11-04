from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.unimodal import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


# Support mosi/mosi_unaligned/mosei/mosei_unaligned/iemocap/iemocap_unaligned
traindata, validdata, testdata = get_dataloader(
    '../affect/processed/mosi_data.pkl')

# mosi
encoders = GRU(20, 50, dropout=True, has_padding=True).cuda()
# encoders=GRU(5,15,dropout=True,has_padding=True).cuda()
# encoders=GRU(300,600,dropout=True,has_padding=True).cuda()
head = MLP(50, 50, 1).cuda()
# mosei/iemocap
'''
encoders=GRU(35,70,dropout=True,has_padding=True).cuda()
encoders=GRU(74,150,dropout=True,has_padding=True).cuda()
encoders=GRU(300,600,dropout=True,has_padding=True).cuda()
head=MLP(820,400,1).cuda()
'''

# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei
train(encoders, head, traindata, validdata, 1000, True, True, task="regression",
      optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01, criterion=torch.nn.L1Loss(), modalnum=0)
# iemocap
'''
train(encoders,head,traindata,validdata,1000,True,True,\
    optimtype=torch.optim.AdamW,lr=1e-4,weight_decay=0.01,modalnum=0)
'''

print("Testing:")
encoder = torch.load('encoder.pt').cuda()
head = torch.load('head.pt').cuda()
test(encoder, head, testdata, True, "regression", 0)
# test(encoder,head,testdata,True,modalnum=0)
