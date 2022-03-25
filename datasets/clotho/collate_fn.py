#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements the collation function for CLOTHO data."""
from typing import MutableSequence, Union, Tuple, AnyStr
from numpy import ndarray

from torch import cat as pt_cat, zeros as pt_zeros, \
    ones as pt_ones, from_numpy, Tensor

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['clotho_collate_fn']


def clotho_collate_fn(batch: MutableSequence[ndarray],
                      nb_t_steps: Union[AnyStr, Tuple[int, int]],
                      input_pad_at: str,
                      output_pad_at: str) \
        -> Tuple[Tensor, Tensor]:
    """Pads data.

    :param batch: Batch data.
    :type batch: list[numpy.ndarray]
    :param nb_t_steps: Number of time steps to\
                       pad/truncate to. Cab use\
                       'max', 'min', or exact number\
                       e.g. (1024, 10).
    :type nb_t_steps: str|(int, int)
    :param input_pad_at: Pad input at the start or\
                         at the end?
    :type input_pad_at: str
    :param output_pad_at: Pad output at the start or\
                          at the end?
    :type output_pad_at: str
    :return: Padded data.
    :rtype: torch.Tensor, torch.Tensor
    """
    if type(nb_t_steps) == str:
        truncate_fn = max if nb_t_steps.lower() == 'max' else min
        in_t_steps = truncate_fn([i[0].shape[0] for i in batch])
        out_t_steps = truncate_fn([i[1].shape[0] for i in batch])
    else:
        in_t_steps, out_t_steps = nb_t_steps

    in_dim = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]

    input_tensor, output_tensor = [], []

    for in_b, out_b in batch:
        if in_t_steps >= in_b.shape[0]:
            padding = pt_zeros(in_t_steps - in_b.shape[0], in_dim).float()
            data = [from_numpy(in_b).float()]
            if input_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)
            tmp_in: Tensor = pt_cat(data)
        else:
            tmp_in: Tensor = from_numpy(in_b[:in_t_steps, :]).float()
        input_tensor.append(tmp_in.unsqueeze_(0))

        if out_t_steps >= out_b.shape[0]:
            padding = pt_ones(out_t_steps - len(out_b)).mul(eos_token).long()
            data = [from_numpy(out_b).long()]
            if output_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)

            tmp_out: Tensor = pt_cat(data)
        else:
            tmp_out: Tensor = from_numpy(out_b[:out_t_steps]).long()
        output_tensor.append(tmp_out.unsqueeze_(0))

    input_tensor = pt_cat(input_tensor)
    output_tensor = pt_cat(output_tensor)

    return input_tensor, output_tensor

# EOF
