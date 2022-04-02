"""Extraneous methods for robotics dataset work."""
import torch
import torch.nn as nn
import numpy as np
import random
import copy
import math
from tqdm import tqdm


def augment_val(val_filename_list, filename_list):
    """Augment lists of filenames so that they match the current directory."""
    filename_list1 = copy.deepcopy(filename_list)
    val_filename_list1 = []

    for name in tqdm(val_filename_list):
        filename = name[:-8]
        found = True

        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        if file_number < 10:
            comp_number = 19
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                filename1 not in val_filename_list1
            ):
                comp_number += -1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number < 0:
                    found = False
                    break
        else:
            comp_number = 0
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                filename1 not in val_filename_list1
            ):
                comp_number += 1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number > 19:
                    found = False
                    break

        if found:
            if filename1 in filename_list1:
                filename_list1.remove(filename1)

            if filename1 not in val_filename_list:
                val_filename_list1.append(filename1)

    val_filename_list1 += val_filename_list

    return val_filename_list1, filename_list1
