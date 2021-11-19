from unimodals.common_models import GRU, MLP
from get_data import get_dataloader
from fusions.common_fusions import ConcatEarly
from training_structures.Simple_Early_Fusion import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '../affect/processed/humor_data.pkl')

# humor 371 81 300
encoders = GRU(752, 1128, dropout=True, has_padding=True).cuda()
head = MLP(1128, 512, 1).cuda()
# encoders=[GRU(35,70,dropout=True,has_padding=True).cuda(), \
#     GRU(74,150,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(820,400,1).cuda()

fusion = ConcatEarly().cuda()

# Support simple early_fusion and early_fusion with removing bias
train(encoders, fusion, head, traindata, validdata, 1000, True, True,
      task="classification", optimtype=torch.optim.AdamW, lr=1e-5, save='humor_ef_best.pt',
      weight_decay=0.01, criterion=torch.nn.MSELoss(), regularization=False)

print("Testing:")
model = torch.load('humor_ef_best.pt').cuda()
test(model, testdata, True, torch.nn.L1Loss(), "classification")
# test(model,testdata,True,)
