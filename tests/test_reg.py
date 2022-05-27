from objective_functions.regularization import *
import torch

def test_perturbation():
    pt = Perturbation()
    assert pt.perturb_tensor(torch.ones(10),1,True).shape == (10,)
    assert pt.get_expanded_logits(torch.ones((10,10)),10).shape == (100,10)

def test_regularization():
    rg = Regularization()
    assert rg.get_batch_statistics(torch.ones(10),10).shape == ()
    assert rg.get_batch_statistics(torch.ones(10),10, 'dif_ent').shape == ()
    assert rg.get_batch_statistics(torch.ones(10),10, 'var').shape == ()
    assert rg.get_batch_norm(torch.ones(10,10),torch.ones(10,)).shape == ()

    assert rg.get_regularization_term(torch.ones(10,10)).shape == ()
    assert rg.get_regularization_term(torch.ones(10,10),optim_method='min_ent').shape == ()
    assert rg.get_regularization_term(torch.ones(10,10),optim_method='max_ent_minus').shape == ()
    fc = torch.nn.Linear(10,2)
    rgl = RegularizationLoss(torch.nn.CrossEntropyLoss,fc, is_pack=False)
    assert callable(rgl)