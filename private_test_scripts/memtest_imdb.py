from objective_functions.recon import recon_weighted_sum, sigmloss1dcentercrop, sigmloss1d
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from training_structures.MFM import train_MFM, test_MFM
from unimodals.common_models import MLP, VGG16, MaxOut_MLP, Linear
from datasets.imdb.get_data import get_dataloader
from fusions.common_fusions import Concat, LowRankTensorFusion, MultiplicativeInteractions2Modal
import torch
import sys
import os
sys.path.append(os.getcwd())


#from training_structures.Contrastive_Learning import train, test
#from training_structures.Simple_Late_Fusion import train, test
#from training_structures.Simple_Early_Fusion import train, test
#from training_structures.cca_onestage import train, test
#from training_structures.unimodal import train, test

# get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(
    '../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)

n_latent = 512

# build encoders, head and fusion layer
encoders = [MaxOut_MLP(512, 512, 300, linear_layer=False).cuda(
), MaxOut_MLP(512, 1024, 4096, 512, False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
#encoders=[MaxOut_MLP(512, 128, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 128, False)]
#encoders=[MaxOut_MLP(512, 512, 300, 256, False), MaxOut_MLP(512, 1024, 4096, 256, False)]
# encoders=None
#encoders=MLP(300, 512, 512).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# encoders=MLP(4096,1024,512).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
decoders = [MLP(n_latent, 600, 300).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(n_latent, 2048, 4096).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
intermediates = [MLP(n_latent, n_latent//2, n_latent//2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
                 MLP(n_latent, n_latent//2, n_latent//2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")), MLP(2*n_latent, n_latent, n_latent//2).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))]
# head=MLP(512,512,23).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# head=Linear(1024,23).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = Linear(256, 23).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# head=Linear(512,23).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#head=MaxOut_MLP(23, 512, 4396).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# refiner=MLP(1024,3072,4396).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
fusion = Concat().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# fusion=LowRankTensorFusion([512,512],512,128).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# fusion=MultiplicativeInteractions2Modal([512,512],1024,'matrix').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
recon_loss = recon_weighted_sum([sigmloss1d, sigmloss1d], [1.0, 1.0])
allmodules = [encoders[0], encoders[1], fusion, head, decoders[0],
              decoders[1], intermediates[0], intermediates[1], intermediates[2]]

print("Training start")
# train


def trainprocess():
    train_MFM(encoders, decoders, head, intermediates, fusion, recon_loss, traindata, validdata, 1000, learning_rate=5e-3,
              savedir="best.pt", task="multilabel", early_stop=True, criterion=torch.nn.BCEWithLogitsLoss())


all_in_one_train(trainprocess, allmodules)


# test
print("Testing: ")
model = torch.load('best.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# encoder=torch.load('encoder.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
# head=torch.load('head.pt').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


def testprocess():
    test_MFM(model, testdata, task="multilabel")


    #test(encoder,head,testdata,task="multilabel", modalnum=0)
all_in_one_test(testprocess, [encoders[0],
                encoders[1], fusion, head, intermediates[-1]])
