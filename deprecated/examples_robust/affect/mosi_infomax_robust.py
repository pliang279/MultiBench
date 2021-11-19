from unimodals.common_models import GRU, MLP
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append('/home/pliang/multibench/MultiBench/datasets/affect')


traindata, validdata, robust_text, robust_vision, robust_audio, robust_timeseries = get_dataloader(
    '../../../affect/processed/mosi_data.pkl', '../../../affect/mosi', 'mosi')

encoders = [GRU(20, 50, dropout=True, has_padding=True).cuda(),
            GRU(5, 15, dropout=True, has_padding=True).cuda(),
            GRU(300, 600, dropout=True, has_padding=True).cuda()]
head = MLP(665, 300, 1).cuda()
fusion = Concat().cuda()


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 1000, True, True,
          task="regression", optimtype=torch.optim.AdamW, lr=1e-4, save=filename,
          weight_decay=0.01, criterion=torch.nn.L1Loss(), regularization=False)


filename = general_train(trainprocess, 'mosi_infomax')


def testprocess(model, robustdata):
    return test(model, robustdata, True, torch.nn.L1Loss(), "regression")


general_test(testprocess, filename, [
             robust_text, robust_vision, robust_audio, robust_timeseries])
