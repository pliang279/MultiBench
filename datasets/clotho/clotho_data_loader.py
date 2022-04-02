#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements the creation of the dataloader for CLOTHO."""

from typing import Callable, Union, Tuple, AnyStr, Optional
from functools import partial
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from .clotho_dataset import ClothoDataset
from .collate_fn import clotho_collate_fn

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['get_clotho_loader']


def get_clotho_loader(data_dir: Path,
                      split: str,
                      input_field_name: str,
                      output_field_name: str,
                      load_into_memory: bool,
                      batch_size: int,
                      nb_t_steps_pad: Union[AnyStr, Tuple[int, int]],
                      shuffle: Optional[bool] = True,
                      drop_last: Optional[bool] = True,
                      input_pad_at: Optional[str] = 'start',
                      output_pad_at: Optional[str] = 'end',
                      num_workers: Optional[int] = 1) \
        -> DataLoader:
    """Gets the clotho data loader.

    :param data_dir: Directory with data.
    :type data_dir: pathlib.Path
    :param split: Split to use (i.e. 'development', 'evaluation')
    :type split: str
    :param input_field_name: Field name of the clotho data\
                             to be used as input data to the\
                             method.
    :type input_field_name: str
    :param output_field_name: Field name of the clotho data\
                             to be used as output data to the\
                             method.
    :type output_field_name: str
    :param load_into_memory: Load all data into memory?
    :type load_into_memory: bool
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param nb_t_steps_pad: Number of time steps to\
                           pad/truncate to. Cab use\
                           'max', 'min', or exact number\
                           e.g. (1024, 10).
    :type nb_t_steps_pad: str|(int, int)
    :param shuffle: Shuffle examples? Defaults to True.
    :type shuffle: bool, optional
    :param drop_last: Drop the last examples if not making\
                      a batch of `batch_size`? Defaults to True.
    :type drop_last: bool, optional
    :param input_pad_at: Pad input at the start or\
                         at the end?
    :type input_pad_at: str
    :param output_pad_at: Pad output at the start or\
                          at the end?
    :type output_pad_at: str
    :param num_workers: Amount of workers, defaults to 1.
    :type num_workers: int, optional
    :return: Dataloader for Clotho data.
    :rtype: torch.utils.data.dataloader.DataLoader
    """
    dataset: ClothoDataset = ClothoDataset(
        data_dir=data_dir, split=split,
        input_field_name=input_field_name,
        output_field_name=output_field_name,
        load_into_memory=load_into_memory)

    collate_fn: Callable = partial(
        clotho_collate_fn,
        nb_t_steps=nb_t_steps_pad,
        input_pad_at=input_pad_at,
        output_pad_at=output_pad_at)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, collate_fn=collate_fn)

# EOF
