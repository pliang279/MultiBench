# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseDragonSuit(Armour):
    pass


class DragonScales(BaseDragonSuit):

    def __init__(self):
        super().__init__('dragon scales', weight=40, armour_class=3, material=M.Dragon)
