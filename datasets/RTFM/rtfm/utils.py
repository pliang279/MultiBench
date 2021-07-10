# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import revtok


def tokenize(sent):
    return [w.strip() for w in revtok.tokenize(sent.lower())]


def get_all_subclasses(c):
    ret = []
    sub = c.__subclasses__()
    for cc in sub:
        ret += get_all_subclasses(cc)
    return ret + sub



