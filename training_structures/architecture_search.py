"""Implements training procedure for MFAS."""
from utils.AUPRC import AUPRC
import torch
import torch.optim as op
import numpy as np

import utils.surrogate as surr
import utils.search_tools as tools

import fusions.searchable as avm
# from eval_scripts.performance import AUPRC
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from tqdm import tqdm


def train(unimodal_files, rep_size, classes, sub_sizes, train_data, valid_data, surrogate, max_labels,
          batch_size=32, epochs=3,
          search_iter=3, num_samples=15, epoch_surrogate=50,
          eta_max=0.001, eta_min=0.000001, Ti=1, Tm=2,
          temperature_init=10.0, temperature_final=0.2, temperature_decay=4.0, max_progression_levels=4,
          lr_surrogate=0.001, use_weightsharing=False):
    """Train MFAS Model.
    
    See https://github.com/slyviacassell/_MFAS/blob/master/models/searchable.py for more details.

    Args:
        unimodal_files (list[dict]): Dictionary of names of files containing pretrained unimodal encoders
        rep_size (int): Size of Representation
        classes (int): Output Size
        sub_sizes (list of tuples): The output size of each layer within the unimodal encoders
        train_data (torch.utils.data.DataLoader): Training data loader
        valid_data (torch.utils.data.DataLoader): Validation data loader
        surrogate (nn.Module): Surrogate Instance
        max_labels (tuple): Search space of input architecture
        batch_size (int, optional): Batch size Defaults to 32.
        epochs (int, optional): Epoch count. Defaults to 3.
        search_iter (int, optional): Number of iterations to search with MFAS. Defaults to 3.
        num_samples (int, optional): Sample number. Defaults to 15.
        epoch_surrogate (int, optional): Surrogate epoch. Defaults to 50.
        eta_max (float, optional): See MFAS github for more details. Defaults to 0.001.
        eta_min (float, optional): See MFAS github for more details. Defaults to 0.000001.
        Ti (int, optional): See MFAS github for more details. Defaults to 1.
        Tm (int, optional): See MFAS github for more details. Defaults to 2.
        temperature_init (float, optional): See MFAS github for more details. Defaults to 10.0.
        temperature_final (float, optional): See MFAS github for more details. Defaults to 0.2.
        temperature_decay (float, optional): See MFAS github for more details. Defaults to 4.0.
        max_progression_levels (int, optional): See MFAS github for more details. Defaults to 4.
        lr_surrogate (float, optional): Surrogate learning rate. Defaults to 0.001.
        use_weightsharing (bool, optional): Use weight-sharing when training architectures for evaluation. Defaults to False.

    Returns:
        _type_: _description_
    """
    searcher = ModelSearcher(train_data, valid_data, search_iter, num_samples, epoch_surrogate,
                             temperature_init, temperature_final, temperature_decay, max_progression_levels, lr_surrogate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    searcher.device = device
    s_data = searcher.search(surrogate,
                             use_weightsharing, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs, max_labels,
                             eta_max, eta_min, Ti, Tm)
    return s_data


class ModelSearcher():
    """Implements MFAS Procedure.
    
    See https://github.com/slyviacassell/_MFAS/blob/master/models/searchable.py for more details.
    """
    
    def __init__(self, train_data, valid_data, search_iter, num_samples, epoch_surrogate, temperature_init, temperature_final, temperature_decay, max_progression_levels, lr_surrogate, device="cuda"):
        """Initialize ModelSearcher Object.

        Args:
            train_data (torch.utils.data.DataLoader): Training Data Dataloader
            valid_data (torch.utils.data.DataLoader): Validation Data Dataloader
            search_iter (int): Number of search iterations
            num_samples (int): Number of samples
            epoch_surrogate (int): Number of epochs per surrogate
            temperature_init (float): Initial softmax temperature
            temperature_final (float): Final softmax temperature
            temperature_decay (float): Softmax temperature decay rate
            max_progression_levels (int): Maximum number of progression levels.
            lr_surrogate (float): Surrogate learning rate.
            device (str, optional): Device to place computation on. Defaults to "cuda".
        """
        self.search_iterations = search_iter
        self.num_samples = num_samples
        self.surrep = epoch_surrogate
        self.tinit = temperature_init
        self.tfinal = temperature_final
        self.tdecay = temperature_decay
        self.max_prog = max_progression_levels
        self.lr_surrogate = lr_surrogate
        self.dataloaders = {'train': train_data, 'dev': valid_data}

        self.device = device

    def search(self, surrogate,
               use_weightsharing, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs, max_labels,
               eta_max, eta_min, Ti, Tm, criterion=torch.nn.MSELoss()):
        """Search for the best model using MFAS.

        Args:
            surrogate (nn.Module): Surrogate cost function module.
            use_weightsharing (bool): Whether to use weight sharing or not.
            unimodal_files (list): List of unimodal encoders, pre-trained.
            rep_size (int): Dimensionality of unimodal encoder output
            classes (int): Number of classes
            sub_sizes (int): Sub sizes
            batch_size (int): Batch size
            epochs (int): Number of epochs
            max_labels (int): Maximum number of labels
            eta_max (float): eta_max for LRCosineAnnealing Scheduler
            eta_min (float): eta_min for LRCosineAnnealingScheduler
            Ti (float): Ti for LRCosineAnnealingScheduler
            Tm (float): Tm for LRCosineAnnealingScheduler
            criterion (nn.Module, optional): Loss function. Defaults to torch.nn.MSELoss().

        Returns:
            torch.Tensor: Surrogate function training data ( i.e. model configs and their performances )
        """
        searchmethods = {'train_sampled_fun': avm.train_sampled_models,
                         'get_layer_confs': avm.get_possible_layer_configurations}
        surro_dict = {'model': surrogate, 'criterion': criterion}
        return self._epnas(avm.Searchable, surro_dict, self.dataloaders, searchmethods,
                           use_weightsharing, self.device, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs,
                           eta_max, eta_min, Ti, Tm, max_labels)

    def _epnas(self, model_type, surrogate_dict, dataloaders, dataset_searchmethods,
               use_weightsharing, device, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs,
               eta_max, eta_min, Ti, Tm, max_labels):

        # surrogate
        surrogate = surrogate_dict['model']
        s_crite = surrogate_dict['criterion']
        s_data = surr.SurrogateDataloader()
        s_optim = op.Adam(surrogate.parameters(), lr=self.lr_surrogate)

        # search functions that are specific to the dataset
        train_sampled_models = dataset_searchmethods['train_sampled_fun']
        get_possible_layer_configurations = dataset_searchmethods['get_layer_confs']

        temperature = self.tinit

        sampled_k_confs = []

        shared_weights = dict()

        # repeat process search_iterations times
        for si in range(self.search_iterations):

            print("Search iteration {} ".format(si))

            # for each fusion
            for progression_index in range(self.max_prog):

                print("Progressive step {} ".format(progression_index))

                # Step 1: unfold layer (fusion index)
                list_possible_layer_confs = get_possible_layer_configurations(
                    max_labels)

                # Step 2: merge previous top with unfolded configurations
                all_configurations = tools.merge_unfolded_with_sampled(sampled_k_confs, list_possible_layer_confs,
                                                                       progression_index)

                # Step 3: obtain accuracies for all possible unfolded configurations
                # if first execution, just train all, if not, use surrogate to predict them
                if si + progression_index == 0:
                    # the type of each element in all_accuracies is tensor
                    # check avmnist_searchable for details
                    all_accuracies = train_sampled_models(all_configurations, model_type, dataloaders,
                                                          use_weightsharing, device, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs,
                                                          eta_max, eta_min, Ti, Tm,
                                                          state_dict=shared_weights)
                    tools.update_surrogate_dataloader(
                        s_data, all_configurations, all_accuracies)
                    tools.train_surrogate(
                        surrogate, s_data, s_optim, s_crite, self.surrep, device)
                    print("Predicted accuracies: ")
                    print(list(zip(all_configurations, all_accuracies)))

                else:
                    # the type of each element in all_accuracies is numpy.float32
                    # check surrogate.py for details
                    all_accuracies = tools.predict_accuracies_with_surrogate(
                        all_configurations, surrogate, device)
                    print("Predicted accuracies: ")
                    print(list(zip(all_configurations, all_accuracies)))

                # Step 4: sample K architectures and train them.
                # this should happen only if not first iteration because in that case,
                # all confs were trained in step 3
                if si + progression_index == 0:
                    # move tensor from cuda:0 to cpu
                    all_accuracies = [i.cpu() if (not isinstance(i, int)) and i.is_cuda else i for i in all_accuracies]

                    sampled_k_confs = tools.sample_k_configurations(all_configurations, all_accuracies,
                                                                    self.num_samples, temperature)

                    estimated_accuracies = tools.predict_accuracies_with_surrogate(all_configurations, surrogate,
                                                                                   device)
                    diff = np.abs(np.array(estimated_accuracies) -
                                  np.array(all_accuracies))
                    print("Error on accuracies = {}".format(diff))

                else:
                    sampled_k_confs = tools.sample_k_configurations(all_configurations, all_accuracies,
                                                                    self.num_samples, temperature)
                    sampled_k_accs = train_sampled_models(sampled_k_confs, model_type, dataloaders,
                                                          use_weightsharing, device, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs,
                                                          eta_max, eta_min, Ti, Tm,
                                                          state_dict=shared_weights)

                    tools.update_surrogate_dataloader(
                        s_data, sampled_k_confs, sampled_k_accs)
                    err = tools.train_surrogate(
                        surrogate, s_data, s_optim, s_crite, self.surrep, device)

                    print("Trained architectures: ")
                    print(list(zip(sampled_k_confs, sampled_k_accs)))
                    print("with surrogate error: {}".format(err))

                # temperature decays at each step
                iteration = si * self.search_iterations + progression_index
                temperature = tools.compute_temperature(
                    iteration, self.tinit, self.tfinal, self.tdecay)

        return s_data


def single_test(model, test_dataloader, auprc=False):
    """Get accuracy for a single dataloader for MFAS.

    Args:
        model (nn.Module): MFAS Model
        test_dataloader (torch.utils.data.DataLoader): Test dataloader to sample from
        auprc (bool, optional): Whether to get AUPRC scores or not. Defaults to False.

    Returns:
        dict: Dictionary of (metric, value) pairs.
    """
    total = 0
    corrects = 0
    pts = []
    with torch.no_grad():
        for j in test_dataloader:
            x = [y.float().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) for y in j[:-1]]
            out = model(x)
            outs = torch.nn.Softmax()(out)
            for ii in range(len(outs)):
                total += 1
                if outs[ii].tolist().index(max(outs[ii])) == j[-1][ii].item():
                    corrects += 1
                pts.append([outs[ii][1], j[-1][ii].item()])
    print('test acc: '+str(float(corrects)/total))
    if auprc:
        print("AUPRC: "+str(AUPRC(pts)))
    return {'Accuracy': float(corrects)/total}


def test(model, test_dataloaders_all, dataset, method_name='My method', auprc=False, no_robust=False):
    """Test MFAS Model.

    Args:
        model (nn.Module): Module to test.
        test_dataloaders_all (list): List of dataloaders
        dataset (str): Name of dataset.
        method_name (str, optional): Method name. Defaults to 'My method'.
        auprc (bool, optional): Whether to output AUPRC scores or not. Defaults to False.
        no_robust (bool, optional): Whether to not apply robustness transformations or not. Defaults to False.
    """
    if no_robust:
        def _testprocess():
            single_test(model, test_dataloaders_all, auprc)
        all_in_one_test(_testprocess, [model])
        return

    def _testprocess():
        single_test(model, test_dataloaders_all[list(
            test_dataloaders_all.keys())[0]][0], auprc)
    all_in_one_test(_testprocess, [model])
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print("Testing on noisy data ({})...".format(noisy_modality))
        robustness_curve = dict()
        for test_dataloader in tqdm(test_dataloaders):
            single_test_result = single_test(model, test_dataloader, auprc)
            for k, v in single_test_result.items():
                curve = robustness_curve.get(k, [])
                curve.append(v)
                robustness_curve[k] = curve
        for measure, robustness_result in robustness_curve.items():
            robustness_key = '{} {}'.format(dataset, noisy_modality)
            print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                relative_robustness(robustness_result, robustness_key))))
            if len(robustness_curve) != 1:
                robustness_key = '{} {}'.format(robustness_key, measure)
            print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                effective_robustness(robustness_result, robustness_key))))
            fig_name = '{}-{}-{}-{}'.format(method_name,
                                            robustness_key, noisy_modality, measure)
            single_plot(robustness_result, robustness_key, xlabel='Noise level',
                        ylabel=measure, fig_name=fig_name, method=method_name)
            print("Plot saved as "+fig_name)
