# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseCrossbow(Weapon):
    pass


class Crossbow(BaseCrossbow):

    def __init__(self):
        super().__init__('crossbow', weight=50, damage=D.Dice.from_str('d2'), material=M.Wood, hit=0)


class CrossbowBolt(BaseCrossbow):

    def __init__(self):
        super().__init__('crossbow bolt', weight=1, damage=D.Dice.from_str('d6+1'), material=M.Iron, hit=0)
