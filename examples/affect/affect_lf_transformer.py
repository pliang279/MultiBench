import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch

from fusions.common_fusions import ConcatEarly
from datasets.affect.get_data import get_dataloader
from unimodals.common_models import Transformer, MLP

from training_structures.Supervised_Learning import train, test

from private_test_scripts.all_in_one import all_in_one_train

# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, _, robust_text, robust_vision, robust_audio, robust_all = \
    get_dataloader('/home/pliang/multibench/affect/processed/mosi_raw.pkl')

# mosi/mosei
encoders = [Transformer().cuda()] * 3

head = MLP(820, 512, 1).cuda()

# humor/sarcasm
# encoders = [Transformer().cuda()] * 3
# head = MLP(1368, 512, 1).cuda()

all_modules = [*encoders, head]

fusion = ConcatEarly().cuda()


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 1000, task="regression", optimtype=torch.optim.AdamW,
        lr=1e-4, save='mosi_lf_best.pt', weight_decay=0.01, objective=torch.nn.L1Loss())

all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('mosi_lf_best.pt').cuda()

test(model, robust_text, True, torch.nn.L1Loss(), "regression")
test(model, robust_vision, True, torch.nn.L1Loss(), "regression")
test(model, robust_audio, True, torch.nn.L1Loss(), "regression")
test(model, robust_all, True, torch.nn.L1Loss(), "regression")



