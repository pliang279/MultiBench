# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseShirt(Armour):
    pass


class HawaiianShirt(BaseShirt):

    def __init__(self):
        super().__init__('Hawaiian shirt', weight=5, armour_class=0, material=M.Cloth)


class TShirt(BaseShirt):

    def __init__(self):
        super().__init__('T-shirt', weight=5, armour_class=0, material=M.Cloth)
