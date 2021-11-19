from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader
from unimodals.common_models import MLP
from training_structures.unimodal import train, test
from fusions.mult import MULTModel
from torch import nn
import torch
import sys
import os
sys.path.append(os.getcwd())

sys.path.append('/home/pliang/multibench/MultiBench/datasets/affect')


traindata, validdata, robust_text, robust_vision, robust_audio, robust_timeseries = get_dataloader(
    '../../../affect/processed/mosei_senti_data.pkl', '../../../affect/mosei', 'mosei')

# mosi
# encoders=GRU(325,512,dropout=True,has_padding=True).cuda()
# head=MLP(512,256, 1).cuda()

# mosei
encoders = MULTModel(3).cuda()
head = nn.Identity()


def trainprocess(filename):
    train(encoders, head, traindata, validdata, 1000, True, True, task="regression", optimtype=torch.optim.AdamW,
          lr=1e-5, save=filename, weight_decay=0.01, criterion=torch.nn.L1Loss(), regularization=False)


filename = general_train(trainprocess, 'affect_mult')


def testprocess(model, robustdata):
    return test(model, robustdata, True, torch.nn.L1Loss(), "regression")


general_test(testprocess, filename, [
             robust_text, robust_vision, robust_audio, robust_timeseries])
