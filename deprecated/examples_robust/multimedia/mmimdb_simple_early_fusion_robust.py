from unimodals.common_models import MLP, VGG16, Linear, MaxOut_MLP
from robustness.all_in_one import general_train, general_test
from get_data_robust import get_dataloader, get_dataloader_robust
from fusions.common_fusions import Concat
from training_structures.Simple_Early_Fusion import train, test
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')

traindata, validdata = get_dataloader(
    '../../../video/multimodal_imdb.hdf5', vgg=True, batch_size=128)
robust_vision, robust_text = get_dataloader_robust(
    '../../../video/mmimdb', batch_size=128)

encoders = None
#encoders=[MLP(300, 512, 512), MLP(4096, 1000, 512)]
#encoders=[MLP(300, 512, 512), VGG16(512)]
head = MaxOut_MLP(23, 512, 4396).cuda()
fusion = Concat().cuda()


def trainprocess(filename):
    train(encoders, fusion, head, traindata, validdata, 1000, early_stop=True, task="multilabel", regularization=False,
          save=filename, optimtype=torch.optim.AdamW, lr=4e-2, weight_decay=0.01, criterion=torch.nn.BCEWithLogitsLoss())


filename = general_train(trainprocess, 'mmimdb_simple_early_fusion')


def testprocess(model, robustdata):
    return test(model, robustdata, criterion=torch.nn.BCEWithLogitsLoss(), task="multilabel")


general_test(testprocess, filename, [
             robust_vision, robust_text], multi_measure=True)
