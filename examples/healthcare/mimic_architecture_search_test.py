#from unimodals.common_models import LeNet, MLP, Constant
import sys
import os

sys.path.append(os.getcwd())

from .training_structures.architecture_search import train, test # noqa
import utils.surrogate as surr # noqa
import torch # noqa
from torch import nn # noqa
from datasets.mimic.get_data import get_dataloader # noqa
from fusions.common_fusions import Concat # noqa


traindata, validdata, testdata = get_dataloader(
    1, imputed_path='datasets/mimic/im.pk')

model = torch.load('temp/best.pt').cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 1', auprc=True)
