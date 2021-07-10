# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseAxe(Weapon):
    pass


class Axe(BaseAxe):

    def __init__(self):
        super().__init__('axe', weight=60, damage=D.Dice.from_str('d4'), material=M.Iron, hit=0)


class BattleAxe(BaseAxe):

    def __init__(self):
        super().__init__('battle-axe', weight=120, damage=D.Dice.from_str('d6+2d4'), material=M.Iron, hit=0)
