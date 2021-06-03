import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch

from training_structures.unimodal import train, test
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
from get_data_robust import get_dataloader, get_dataloader_robust
from robustness.all_in_one import general_train, general_test
from unimodals.common_models import MLP

traindata, validdata = get_dataloader('../../../video/multimodal_imdb.hdf5', vgg=True)
robust_vision, robust_text = get_dataloader_robust('../../../video/mmimdb')

encoders=MLP(300, 512, 512).cuda()
#encoders=MLP(4096,1024,512).cuda()
head=MLP(512,512,23).cuda()

def trainprocess(filename_encoder, filename_head):
    train(encoders,head,traindata,validdata,1000, early_stop=True,task="multilabel", save_encoder=filename_encoder, modalnum=0,\
    save_head=filename_head, optimtype=torch.optim.AdamW,lr=1e-4,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())
filename = general_train(trainprocess, 'mmimdb_unimodal', encoder=True)

def testprocess(encoder, head, robustdata):
    return test(encoder, head, robustdata,task="multilabel", modalnum=0)
general_test(testprocess, filename, [robust_vision, robust_text], multi_measure=True, encoder=True)