from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from unimodals.common_models import GRU, MLP
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # __file__ father path
print(BASE_DIR)
sys.path.append(BASE_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# analysis the time and space complexity

sys.path.append('/home/pliang/multibench/MultiBench/datasets/affect')

# Support mosi/mosi_unaligned/mosei/mosei_unaligned/iemocap/iemocap_unaligned
traindata, validdata, robust_text, robust_vision, robust_audio, robust_timeseries = get_dataloader(
    '../../../affect/processed/mosei_senti_data.pkl', '../../../affect/mosei', 'mosei')

# mosi
# encoders=[GRU(20,50,dropout=True,has_padding=True).cuda(), \
#     GRU(5,15,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(665,300,1).cuda()

# mosei/iemocap

encoders = [GRU(35, 70, dropout=True, has_padding=True).cuda(),
            GRU(74, 150, dropout=True, has_padding=True).cuda(),
            GRU(300, 600, dropout=True, has_padding=True).cuda()]
head = MLP(820, 400, 1).cuda()

# iemocap
'''
encoders=[GRU(35,70,dropout=True,has_padding=True).cuda(), \
    GRU(74,150,dropout=True,has_padding=True).cuda(),\
    GRU(300,600,dropout=True,has_padding=True).cuda()]
head=MLP(820,400,4).cuda()
'''
fusion = Concat().cuda()
allmodules = [encoders[0], encoders[1], encoders[2], head, fusion]
# Support simple late_fusion and late_fusion with removing bias
# Simply change regularization=True
# mosi/mosei


def trainprocess(filename):
    # train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    #     task="regression",optimtype=torch.optim.AdamW,lr=1e-4,save=filename, \
    #     weight_decay=0.01,criterion=torch.nn.L1Loss(),regularization=False)
    train(encoders, fusion, head, traindata, validdata, 1000, True, True,
          task="regression", optimtype=torch.optim.AdamW, lr=1e-4, save=filename,
          weight_decay=0.01, criterion=torch.nn.MSELoss(), regularization=False)


filename = general_train(trainprocess, 'mosi_lf_time_space_est', encoder=True)
# iemocap
'''
train(encoders,fusion,head,traindata,validdata,1000,True,True, \
    optimtype=torch.optim.AdamW,lr=1e-4,save='best.pt', \
    weight_decay=0.01,regularization=False)
'''


def testprocess(model, robustdata):
    return test(model, robustdata, True, torch.nn.L1Loss(), "regression")


general_test(testprocess, filename, [
             robust_text, robust_vision, robust_audio, robust_timeseries])
# test(model,testdata,True,)
