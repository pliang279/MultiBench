import sys
import os
sys.path.append(os.getcwd())

import torch

#from training_structures.Contrastive_Learning import train, test
#from training_structures.Simple_Late_Fusion import train, test
#from training_structures.Simple_Early_Fusion import train, test
#from training_structures.cca_onestage import train, test
from training_structures.unimodal import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import MLP, VGG16, MaxOut_MLP, Linear

from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test
#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader('../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)

#build encoders, head and fusion layer
#encoders=[MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
#encoders=[MaxOut_MLP(512, 128, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 128, False)]
#encoders=[MaxOut_MLP(512, 512, 300, 256, False), MaxOut_MLP(512, 1024, 4096, 256, False)]
#encoders=None
encoders=MLP(300, 512, 512).cuda()
#encoders=MLP(4096,1024,512).cuda()
head=MLP(512,512,23).cuda()
#head=Linear(1024,23).cuda()
#head=Linear(256,23).cuda()
#head=Linear(512,23).cuda()
#head=MaxOut_MLP(23, 512, 4396).cuda()
#refiner=MLP(1024,3072,4396).cuda()
fusion=Concat().cuda()
allmodules = [encoders,fusion,head,]

print("Training start")
#train
def trainprocess(): 
    train(encoders,head,traindata,validdata,1000, early_stop=True,task="multilabel", save_encoder="encoder.pt", modalnum=0,\
        save_head="head.pt", optimtype=torch.optim.AdamW,lr=1e-4,weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())
all_in_one_train(trainprocess,allmodules)


#test
print("Testing: ")
#model=torch.load('best.pt').cuda()
encoder=torch.load('encoder.pt').cuda()
head=torch.load('head.pt').cuda()
def testprocess():
    #test(model,testdata,criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
    test(encoder,head,testdata,task="multilabel", modalnum=0)
all_in_one_test(testprocess,[encoder, head])