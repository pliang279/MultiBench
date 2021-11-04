from unimodals.common_models import GRU, MLP
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader
from fusions.common_fusions import Concat
from training_structures.unimodal import train, test
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append('/home/pliang/multibench/MultiBench/datasets/affect')

# Support mosi/mosi_unaligned/mosei/mosei_unaligned/iemocap/iemocap_unaligned
traindata, validdata, robust_text, robust_vision, robust_audio, robust_timeseries = get_dataloader(
    '../../../affect/processed/mosi_data.pkl', '../../../affect/mosi', 'mosi')

# mosi
encoders = GRU(20, 50, dropout=True, has_padding=True).cuda()
# encoders=GRU(5,15,dropout=True,has_padding=True).cuda()
# encoders=GRU(300,600,dropout=True,has_padding=True).cuda()
head = MLP(50, 50, 1).cuda()
# mosei/iemocap
'''
encoders=GRU(35,70,dropout=True,has_padding=True).cuda()
encoders=GRU(74,150,dropout=True,has_padding=True).cuda()
encoders=GRU(300,600,dropout=True,has_padding=True).cuda()
head=MLP(820,400,1).cuda()
'''

# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei


def trainprocess(filename_encoder, filename_head):
    train(encoders, head, traindata, validdata, 1000, True, True, task="regression",
          optimtype=torch.optim.AdamW, lr=1e-4, weight_decay=0.01, criterion=torch.nn.L1Loss(), save_encoder=filename_encoder, modalnum=0, save_head=filename_head)


filename = general_train(trainprocess, 'mosi_unimodal', encoder=True)
# iemocap
'''
train(encoders,head,traindata,validdata,1000,True,True,\
    optimtype=torch.optim.AdamW,lr=1e-4,weight_decay=0.01,modalnum=0)
'''


def testprocess(encoder, head, robustdata):
    return test(encoder, head, robustdata, True, "regression", 0)


general_test(testprocess, filename, [
             robust_text, robust_vision, robust_audio, robust_timeseries], encoder=True)
# test(encoder,head,testdata,True,modalnum=0)
