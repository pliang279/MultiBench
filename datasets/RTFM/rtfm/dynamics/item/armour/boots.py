# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseBoot(Armour):
    pass


class LowBoots(BaseBoot):

    def __init__(self):
        super().__init__('low boots', weight=10, armour_class=1, material=M.Leather)


class HighBoots(BaseBoot):

    def __init__(self):
        super().__init__('high boots', weight=20, armour_class=2, material=M.Leather)


class IronShoes(BaseBoot):

    def __init__(self):
        super().__init__('iron shoes', weight=50, armour_class=2, material=M.Iron)
