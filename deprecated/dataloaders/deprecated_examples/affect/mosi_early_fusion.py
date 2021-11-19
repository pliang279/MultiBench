from unimodals.common_models import GRU, MLP
from get_data import get_dataloader
from fusions.common_fusions import ConcatEarly
from training_structures.Simple_Early_Fusion import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


# Support mosi/mosi_unaligned/mosei/mosei_unaligned
traindata, validdata, testdata = get_dataloader(
    '../affect/processed/mosei_senti_data.pkl')

# mosi
# encoders=GRU(325,512,dropout=True,has_padding=True).cuda()
# head=MLP(512,256, 1).cuda()

# mosei
encoders = GRU(409, 800, dropout=True, has_padding=True).cuda()
head = MLP(800, 400, 1).cuda()
# encoders=[GRU(35,70,dropout=True,has_padding=True).cuda(), \
#     GRU(74,150,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(820,400,1).cuda()
# iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).cuda(), \
    GRU(74,150,dropout=True,has_padding=True).cuda(),\
    GRU(300,600,dropout=True,has_padding=True).cuda()]
head=MLP(820,400,4).cuda()
'''
fusion = ConcatEarly().cuda()

# Support simple early_fusion and early_fusion with removing bias
# mosi/mosei
train(encoders, fusion, head, traindata, validdata, 1000, True, True,
      task="regression", optimtype=torch.optim.AdamW, lr=1e-5, save='mosei_ef_best.pt',
      weight_decay=0.01, criterion=torch.nn.L1Loss(), regularization=False)
# iemocap
'''
train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    optimtype=torch.optim.AdamW,lr=1e-4,save='best.pt', \
    weight_decay=0.01,regularization=False)
'''

print("Testing:")
model = torch.load('mosei_ef_best.pt').cuda()
test(model, testdata, True, torch.nn.L1Loss(), "regression")
# test(model,testdata,True,)
