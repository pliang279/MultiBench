import sys
import os
sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from training_structures.Supervised_Learning import train, test
from unimodals.common_models import GRU, MLP
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import Concat
import torch



# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = get_dataloader(
    '/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', robust_test=False, data_type='mosi')

# traindata, validdata, test_robust = \
#     get_dataloader('/home/pliang/multibench/affect/sarcasm.pkl', robust_test=False)

# mosi/mosei
encoders = [GRU(35, 70, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(74, 200, dropout=True, has_padding=True, batch_first=True).cuda(),
            GRU(300, 600, dropout=True, has_padding=True, batch_first=True).cuda()]
head = MLP(870, 870, 1).cuda()

# encoders=[GRU(20,40,dropout=True,has_padding=True).cuda(), \
#     GRU(5, 20,dropout=True,has_padding=True).cuda(),\
#     GRU(300, 600,dropout=True,has_padding=True).cuda()]
# head=MLP(660,512,1, dropoutp=True).cuda()

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=False, batch_first=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=False, batch_first=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=False, batch_first=True).cuda()]
# head=MLP(1368,512,1).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
      early_stop=False, is_packed=True, lr=1e-3, save='mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

print("Testing:")
model = torch.load('mosi_lf_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
