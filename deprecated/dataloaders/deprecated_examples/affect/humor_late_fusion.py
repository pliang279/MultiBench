from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Supervised_Learning import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


# Support mosi/mosi_unaligned/mosei/mosei_unaligned
traindata, validdata, testdata = get_dataloader(
    '/home/pliang/multibench/affect/processed/humor_data.pkl')

# humor 371 81 300
encoders = GRU(752, 1128, dropout=True, has_padding=True).cuda()
head = MLP(1128, 512, 1).cuda()
# encoders=[GRU(35,70,dropout=True,has_padding=True).cuda(), \
#     GRU(74,150,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(820,400,1).cuda()

fusion = Concat().cuda()

# Support simple late_fusion and late_fusion with removing bias
train(encoders, fusion, head, traindata, validdata, 1000, is_packed=True, early_stop=True,
      task="classification", optimtype=torch.optim.AdamW, lr=1e-5, save='humor_lf_best.pt',
      weight_decay=0.01, objective=torch.nn.MSELoss())

print("Testing:")
model = torch.load('humor_lf_best.pt').cuda()
test(model, testdata, True, torch.nn.L1Loss(), "regression")
# test(model,testdata,True,)
