#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implements the CLOTHO dataset as a torch dataset."""
from typing import Tuple, List, AnyStr, Union
from pathlib import Path

from numpy import ndarray, recarray
from torch.utils.data import Dataset
from numpy import load as np_load

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


class ClothoDataset(Dataset):
    """Implements the CLOTHO dataset as a torch dataset."""
    def __init__(self, data_dir: Path,
                 split: AnyStr,
                 input_field_name: AnyStr,
                 output_field_name: AnyStr,
                 load_into_memory: bool) \
            -> None:
        """Initialization of a Clotho dataset object.

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
        """
        super(ClothoDataset, self).__init__()
        if split == 'evaluation':
            splitstr = 'clotho_dataset_eva'
        else:
            splitstr = 'clotho_dataset_dev'
        the_dir: Path = data_dir.joinpath(splitstr)

        self.examples: List[Path] = sorted(the_dir.iterdir())
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory

        if load_into_memory:
            self.examples: List[recarray] = [np_load(str(f), allow_pickle=True)
                                             for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values.
        :rtype: numpy.ndarray. numpy.ndarray
        """
        ex: Union[Path, recarray] = self.examples[item]
        if not self.load_into_memory:
            ex: recarray = np_load(str(ex), allow_pickle=True)

        in_e, ou_e = [ex[i].item()
                      for i in [self.input_name, self.output_name]]

        return in_e, ou_e

# EOF
