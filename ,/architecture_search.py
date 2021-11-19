from utils.AUPRC import AUPRC
import torch
import torch.optim as op
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset

import utils.surrogate as surr
import utils.search_tools as tools

import fusions.searchable as avm


def train(unimodal_files, rep_size, classes, sub_sizes, train_data, valid_data, surrogate, max_labels,
          batch_size=32, epochs=3,
          search_iter=3, num_samples=15, epoch_surrogate=50,
          eta_max=0.001, eta_min=0.000001, Ti=1, Tm=2,
          temperature_init=10.0, temperature_final=0.2, temperature_decay=4.0, max_progression_levels=4,
          lr_surrogate=0.001, use_weightsharing=False):
    searcher = ModelSearcher(train_data, valid_data, search_iter, num_samples, epoch_surrogate,
                             temperature_init, temperature_final, temperature_decay, max_progression_levels, lr_surrogate)
    s_data = searcher.search(surrogate,
                             use_weightsharing, unimodal_files, rep_size, classes, sub_sizes, batch_size, epochs, max_labels,
                             eta_max, eta_min, Ti, Tm)
    return s_data


class ModelSearcher():
    def __init__(self, train_data, valid_data, search_iter, num_samples, epoch_surrogate, temperature_init, temperature_final, temperature_decay, max_progression_levels, lr_surrogate, device="cuda"):
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
                    all_accuracies = [i.cpu() for i in all_accuracies]

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


def test(model, test_dataloader, auprc=False):
    total = 0
    corrects = 0
    pts = []
    with torch.no_grad():
        for j in test_dataloader:
            x = [y.float().cuda() for y in j[:-1]]
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
