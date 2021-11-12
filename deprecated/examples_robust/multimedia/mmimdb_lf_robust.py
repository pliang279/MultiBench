from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader, get_dataloader_robust
from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from fusions.common_fusions import Concat
from training_structures.Simple_Late_Fusion import train, test
import torch
import sys
import os
sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')
sys.path.append('/home/pliang/multibench/MultiBench')

# from get_data_robust import get_dataloader, get_dataloader_robust

traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', batch_size=128, vgg=True)
robustdata = get_dataloader_robust(
    '../../../video/mmimdb', '../../../video/multimodal_imdb.hdf5', batch_size=128)

encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False),
            MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head = Linear(1024, 23).cuda()
fusion = Concat().cuda()


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", regularization=False,
          save=filename, optimtype=torch.optim.AdamW, lr=8e-3, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_lf')


def testprocess(model, testdata):
    return test(model, testdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")


general_test(testprocess, filename, robustdata, multi_measure=True)

#traindata, validdata, testdata = get_dataloader('../../../video/multimodal_imdb.hdf5', batch_size=1, vgg=True)

#encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
##encoders=[MLP(300, 512, 512), VGG16(512)]
# head=MLP(1024,512,23).cuda()
# fusion=Concat().cuda()

# train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel", regularization=False,\
#    save="best_lf.pt", optimtype=torch.optim.AdamW,lr=5e-5,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


# model=torch.load('best_lf.pt').cuda()
# test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
