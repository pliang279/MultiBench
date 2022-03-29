import torch
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from private_test_scripts.all_in_one import all_in_one_train # noqa
from training_structures.Supervised_Learning import train, test # noqa
from unimodals.common_models import Transformer, MLP # noqa
from datasets.affect.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa

# mosi_data.pkl, mosei_senti_data.pkl
# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, test_robust = \
    get_dataloader('/home/paul/MultiBench/mosi_data.pkl', robust_test=False)

# mosi/mosei
encoders = [Transformer(20, 40).cuda(),
            Transformer(5, 10).cuda(),
            Transformer(300, 600).cuda()]
head = MLP(650, 256, 1).cuda()

# humor/sarcasm
# encoders=[Transformer(371,400).cuda(), \
#     Transformer(81,100).cuda(),\
#     Transformer(300,600).cuda()]
# head=MLP(1100,256,1).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW,
      early_stop=True, is_packed=True, lr=1e-4, save='mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


print("Testing:")
model = torch.load('mosi_lf_best.pt').cuda()

test(model=model, test_dataloaders_all=test_robust, dataset='mosi', is_packed=True,
     criterion=torch.nn.L1Loss(), task='posneg-classification', no_robust=True)
