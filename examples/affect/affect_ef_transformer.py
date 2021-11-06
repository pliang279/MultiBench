from unimodals.common_models import Transformer, MLP, Sequential, Identity
from private_test_scripts.all_in_one import all_in_one_train
from training_structures.Supervised_Learning import train, test
from datasets.affect.get_data import get_dataloader
from fusions.common_fusions import ConcatEarly
import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
# raw_path: mosi.hdf5, mosei.hdf5, sarcasm_raw_text.pkl, humor_raw_text.pkl
traindata, validdata, testdata = get_dataloader(
    '/home/paul/MultiBench/mosei_senti_data.pkl', robust_test=False)

# mosi/mosei
encoders = [Identity().cuda(), Identity().cuda(), Identity().cuda()]
head = Sequential(Transformer(409, 300).cuda(), MLP(300, 128, 1)).cuda()

# humor/sarcasm
# encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
# head = Sequential(Transformer(752, 300).cuda(),MLP(300, 128, 1)).cuda()


fusion = ConcatEarly().cuda()

<<<<<<< HEAD
train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=True, early_stop=True,lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())
=======
train(encoders, fusion, head, traindata, validdata, 100, task="regression", optimtype=torch.optim.AdamW, is_packed=True, early_stop=True,
      lr=1e-4, save='mosi_ef_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

# all_in_one_train(trainprocess, all_modules)
>>>>>>> 63e65eaa7cf85f2b73ee4e4da104de5a433c2894

print("Testing:")
model = torch.load('mosi_ef_best.pt').cuda()
test(model, testdata, 'affect', is_packed=True,
     criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
