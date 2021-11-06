from unimodals.common_models import GRUWithLinear, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Contrastive_Learning import train, test
import torch
import sys
import os
sys.path.append(os.getcwd())


traindata, validdata, testdata = get_dataloader(
    '../affect/processed/mosi_data.pkl', 100,)

'''
encoders=[GRU(20,50,dropout=True,has_padding=True).cuda(), \
    GRU(5,15,dropout=True,has_padding=True).cuda(),\
    GRU(300,600,dropout=True,has_padding=True).cuda()]
'''
encoders = [GRUWithLinear(20, 50, 200, dropout=True, has_padding=True).cuda(),
            GRUWithLinear(300, 600, 200, dropout=True, has_padding=True).cuda()]
head = MLP(200, 100, 1).cuda()
fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 1000, True, True,
      optimtype=torch.optim.AdamW, lr=1e-4, save='best_contrast.pt',
      weight_decay=0.01)

print("Testing:")
model = torch.load('best_contrast.pt').cuda()
test(model, testdata, True,)
