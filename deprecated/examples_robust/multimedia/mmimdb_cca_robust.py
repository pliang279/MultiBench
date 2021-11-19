from robustness.all_in_one import general_train, general_test
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from get_data_robust import get_dataloader, get_dataloader_robust
from fusions.common_fusions import Concat
from training_structures.cca_onestage import train, test
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')

traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', batch_size=800, vgg=True)
robustdata = get_dataloader_robust(
    '../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5', batch_size=800)

outdim = 256
encoders = [MaxOut_MLP(512, 512, 300, outdim, False),
            MaxOut_MLP(512, 1024, 4096, outdim, False)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
#encoders=[Linear(300, 512), Linear(4096,512)]
head = Linear(2*outdim, 23).cuda()
fusion = Concat().cuda()


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", outdim=outdim,
          save=filename, optimtype=torch.optim.AdamW, lr=1e-2, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_cca')


def testprocess(model, testdata):
    return test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")


general_test(testprocess, filename, robustdata, multi_measure=True)
