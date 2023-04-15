"""Implements various objectives for supervised learning objectives."""
import torch
from objective_functions.recon import recon_weighted_sum, elbo_loss
from objective_functions.cca import CCALoss
from objective_functions.regularization import RegularizationLoss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _criterioning(pred, truth, criterion):
    """Handle criterion ideosyncracies."""
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        truth = truth.squeeze() if len(truth.shape) == len(pred.shape) else truth
        return criterion(pred, truth.long().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    if isinstance(criterion, (torch.nn.modules.loss.BCEWithLogitsLoss, torch.nn.MSELoss, torch.nn.L1Loss)):
        return criterion(pred, truth.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))


def MFM_objective(ce_weight, modal_loss_funcs, recon_weights, input_to_float=True, criterion=torch.nn.CrossEntropyLoss()):
    """Define objective for MFM.
    
    :param ce_weight: the weight of simple supervised loss
    :param model_loss_funcs: list of functions that takes in reconstruction and input of each modality and compute reconstruction loss
    :param recon_weights: list of float values indicating the weight of reconstruction loss of each modality
    :param criterion: the loss function for supervised loss (default CrossEntropyLoss)
    """
    recon_loss_func = recon_weighted_sum(modal_loss_funcs, recon_weights)

    def _actualfunc(pred, truth, args):
        ints = args['intermediates']
        reps = args['reps']
        fused = args['fused']
        decoders = args['decoders']
        inps = args['inputs']
        recons = []
        for i in range(len(reps)):
            recons.append(decoders[i](
                torch.cat([ints[i](reps[i]), fused], dim=1)))
        ce_loss = _criterioning(pred, truth, criterion)
        if input_to_float:
            inputs = [i.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for i in inps]
        else:
            inputs = [i.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for i in inps]
        recon_loss = recon_loss_func(recons, inputs)
        return ce_loss*ce_weight+recon_loss
    return _actualfunc


def _reparameterize(mu, logvar, training):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def MVAE_objective(ce_weight, modal_loss_funcs, recon_weights, input_to_float=True, annealing=1.0, criterion=torch.nn.CrossEntropyLoss()):
    """Define objective for MVAE.
    
    :param ce_weight: the weight of simple supervised loss
    :param model_loss_funcs: list of functions that takes in reconstruction and input of each modality and compute reconstruction loss
    :param recon_weights: list of float values indicating the weight of reconstruction loss of each modality
    :param input_to_float: boolean deciding if we should convert input to float or not.
    :param annealing: the annealing factor, i.e. the weight of kl.
    :param criterion: the loss function for supervised loss (default CrossEntropyLoss)
    """
    recon_loss_func = elbo_loss(modal_loss_funcs, recon_weights, annealing)

    def _allnonebuti(i, item):
        ret = [None for w in modal_loss_funcs]
        ret[i] = item
        return ret

    def _actualfunc(pred, truth, args):
        training = args['training']
        reps = args['reps']
        fusedmu, fusedlogvar = args['fused']
        decoders = args['decoders']
        inps = args['inputs']
        reconsjoint = []

        if input_to_float:
            inputs = [i.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for i in inps]
        else:
            inputs = [i.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for i in inps]
        for i in range(len(inps)):
            reconsjoint.append(decoders[i](
                _reparameterize(fusedmu, fusedlogvar, training)))
        total_loss = recon_loss_func(reconsjoint, inputs, fusedmu, fusedlogvar)
        for i in range(len(inps)):
            mu, logvar = reps[i]
            recon = decoders[i](_reparameterize(mu, logvar, training))
            total_loss += recon_loss_func(_allnonebuti(i, recon),
                                          _allnonebuti(i, inputs[i]), mu, logvar)
        total_loss += ce_weight * _criterioning(pred, truth, criterion)
        return total_loss
    return _actualfunc



def CCA_objective(out_dim, cca_weight=0.001, criterion=torch.nn.CrossEntropyLoss()):
    """
    Define loss function for CCA.
    
    :param out_dim: output dimension
    :param cca_weight: weight of cca loss
    :param criterion: criterion for supervised loss
    """
    lossfunc = CCALoss(out_dim, False, device=device)

    def _actualfunc(pred, truth, args):
        ce_loss = _criterioning(pred, truth, criterion)
        outs = args['reps']
        cca_loss = lossfunc(outs[0], outs[1])
        return cca_loss * cca_weight + ce_loss
    return _actualfunc



def RefNet_objective(ref_weight, criterion=torch.nn.CrossEntropyLoss(), input_to_float=True):
    """
    Define loss function for RefNet.
    
    :param ref_weight: weight of refiner loss
    :param criterion: criterion for supervised loss
    :param input_to_float: whether to convert input to float or not
    """
    ss_criterion = torch.nn.CosineEmbeddingLoss()

    def _actualfunc(pred, truth, args):
        ce_loss = _criterioning(pred, truth, criterion)
        refiner = args['refiner']
        fused = args['fused']
        inps = args['inputs']
        refinerout = refiner(fused)
        if input_to_float:
            inputs = [torch.flatten(t, start_dim=1).float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                      for t in inps]
        else:
            inputs = [torch.flatten(t, start_dim=1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for t in inps]

        inputsizes = [t.size(1) for t in inputs]
        ss_loss = 0.0
        loc = 0
        for i in range(len(inps)):
            out = refinerout[:, loc:loc+inputsizes[i]]
            loc += inputsizes[i]
            ss_loss += ss_criterion(out,
                                    inputs[i], torch.ones(out.size(0)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        return ce_loss + ss_loss*ref_weight
    return _actualfunc


def RMFE_object(reg_weight=1e-10, criterion=torch.nn.BCEWithLogitsLoss(), is_packed=False):
    """
    Define loss function for RMFE.
    
    :param model: model used for inference
    :param reg_weight: weight of regularization term
    :param criterion: criterion for supervised loss
    :param is_packed: packed for LSTM or not
    """
    def _regfunc(pred, truth, args):
        model = args['model']
        lossfunc = RegularizationLoss(criterion, model, reg_weight, is_packed)
        ce_loss = _criterioning(pred, truth, criterion)
        inps = args['inputs']
        try:
            reg_loss = lossfunc(pred, [i.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for i in inps])
        except RuntimeError:
            print("No reg loss for validation")
            reg_loss = 0
        return ce_loss+reg_loss
    return _regfunc
