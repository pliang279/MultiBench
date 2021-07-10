# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseTrident(Weapon):
    pass


class Trident(BaseTrident):

    def __init__(self):
        super().__init__('trident', weight=25, damage=D.Dice.from_str('3d4'), material=M.Iron, hit=0)
