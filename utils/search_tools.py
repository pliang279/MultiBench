#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:42:17 2018
@author: juanma
"""

# general use
import numpy as np
import random

# surrogate related
import utils.surrogate as surr

# %%
"""
 Auxiliary functions for correct exploration of Searchable Models 
"""


def predict_accuracies_with_surrogate(configurations, surrogate, device):
    """Use surrogate to predict the effectiveness of different input configurations.

    Args:
        configurations (list[dict]): List of configurations to search from.
        surrogate (nn.Module): Learned surrogate cost model for configuration space.
        device (string): Device to place computation on.   

    Returns:
        list[float]: Accuracy per configuration.
    """
    # uses surrogate to evaluate input configurations

    accs = []

    for c in configurations:
        accs.append(surrogate.eval_model(c, device))

    return accs


def update_surrogate_dataloader(surrogate_dataloader, configurations, accuracies):
    """Update surrogate dataloader with new configurations.

    Args:
        surrogate_dataloader (SurrogateDataloader): Data Loader for Surrogate Cost Model.  
        configurations (list): List of new configurations to add.
        accuracies (list[float]): List of accuracies to train cost model with.
    """
    for conf, acc in zip(configurations, accuracies):
        surrogate_dataloader.add_datum(conf, acc)


def train_surrogate(surrogate, surrogate_dataloader, surrogate_optimizer, surrogate_criterion, ep, device):
    """Train surrogate model using surrogate dataloader.

    Args:
        surrogate (nn.Module): Surrogate cost model instance.
        surrogate_dataloader (torch.utils.data.DataLoader): Data loader for surrogate instance, mapping configurations to accuracies
        surrogate_optimizer (torch.optim.Optimizer): Optimizer for surrogate parameters
        surrogate_criterion (nn.Module): Loss function for surrogate instance.
        ep (int): Number of epochs.
        device (string): Device to train on.

    Returns:
        float: Loss of surrogate with current training data.
    """
    s_data = surrogate_dataloader.get_data(to_torch=True)
    err = surr.train_simple_surrogate(surrogate, surrogate_criterion,
                                      surrogate_optimizer, s_data,
                                      ep, device)

    return err


def sample_k_configurations(configurations, accuracies_, k, temperature):
    """Sample k configurations from list of configurations and accuracies, based on past performance.

    Args:
        configurations (list): List of configurations.
        accuracies_ (list): List of accuracies for each config.
        k (int): Number of configurations to sample on.
        temperature (float): Temperature for the sample distribution.

    Returns:
        list: List of sampled configurations.
    """
    accuracies = np.array(accuracies_)
    p = accuracies / accuracies.sum()
    powered = pow(p, 1.0 / temperature)
    p = powered / powered.sum()

    indices = np.random.choice(len(configurations), k, replace=False, p=p)
    samples = [configurations[i] for i in indices]

    return samples


def sample_k_configurations_uniform(configurations, k):
    """Sample k configurations uniformly.

    Args:
        configurations (list): List of configurations to sample from.
        k (int): Number of configurations to sample.

    Returns:
        list: List of sampled configurations.
    """
    indices = np.random.choice(len(configurations), k)
    samples = [configurations[i] for i in indices]

    return samples


def merge_unfolded_with_sampled(previous_top_k_configurations, unfolded_configurations, layer):
    """Given a list of top k configurations, and an unfolded single layer configuration, merge them together at the given layer index.

    Args:
        previous_top_k_configurations (list): List of configurations of size (seq_len, 3)
        unfolded_configurations (list): Configuration for a single layer (,3)
        layer (int): Index of layer to add unfolded configuration to.

    Raises:
        ValueError: If there are no top k configurations, layer should be 0.

    Returns:
        list: Merged list of configurations.
    """
    # normally, the outpout configurations are evaluated with the surrogate function

    # unfolded_configurations is a configuration for a single layer, so it does not have seq_len dimension
    # previous_top_k_configurations is composed of configurations of size (seq_len,3)

    merged = []

    if not previous_top_k_configurations:
        # this typically executes at the very first iteration of the sequential exploration
        for unfolded_conf in unfolded_configurations:

            if layer == 0:
                new_conf = np.expand_dims(unfolded_conf, 0)
            else:
                raise ValueError(
                    'merge_unfolded_with_sampled: Something weird is happening. previous_top_k_configurations is None, but layer != 0')

            merged.append(new_conf)
    else:
        # most common pathway of execution: there exist previous configurations
        for prev_conf in previous_top_k_configurations:
            for unfolded_conf in unfolded_configurations:
                new_conf = np.copy(prev_conf)
                if layer < len(prev_conf):
                    new_conf[layer] = unfolded_conf
                else:
                    new_conf = np.concatenate(
                        [prev_conf, np.expand_dims(unfolded_conf, 0)], 0)

                merged.append(new_conf)

    return merged


def sample_k_configurations_directly(k, max_progression_levels, get_possible_layer_configurations_fun):
    """Sample k configurations given a set number of progression_levels.

    Args:
        k (int): Number of configurations to sampl.e
        max_progression_levels (int): Maximum number of sample configurations.
        get_possible_layer_configurations_fun (fn): Function to get layer configurations given some index input.

    Returns:
        list: List of sampled configurations.
    """
    configurations = []

    possible_confs_per_layer = []
    for l in range(max_progression_levels):
        possible_confs_per_layer.append(
            get_possible_layer_configurations_fun(l))

    for sample in range(k):
        num_layers_sample = random.randint(1, max_progression_levels)

        conf = []
        for layer in range(num_layers_sample):
            random_layer_conf = sample_k_configurations_uniform(
                possible_confs_per_layer[l], 1)
            conf.append(random_layer_conf)

        conf = np.array(conf)[:, 0, :]
        configurations.append(conf)

    return configurations


def compute_temperature(iteration, init, final, decay):
    """Compute temperature for a given round of the MFAS procedure.

    Args:
        iteration (int): Iteration index
        init (float): Initial temperature
        final (float): Final temperature
        decay (float): Temperature decay rate

    Returns:
        float: Temperature for this round of MFAS.
    """
    temp = (init - final) * np.exp(
        -(iteration + 1.0) ** 2 / decay ** 2) + final
    return temp

# %%
