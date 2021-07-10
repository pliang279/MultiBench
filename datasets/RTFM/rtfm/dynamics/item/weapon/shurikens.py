# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseShuriken(Weapon):
    pass


class Shuriken(BaseShuriken):

    def __init__(self):
        super().__init__('shuriken', weight=1, damage=D.Dice.from_str('d6'), material=M.Iron, hit=0)
