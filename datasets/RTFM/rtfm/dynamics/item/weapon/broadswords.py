# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseBroadsword(Weapon):
    pass


class Broadsword(BaseBroadsword):

    def __init__(self):
        super().__init__('broadsword', weight=70, damage=D.Dice.from_str('d6+1'), material=M.Iron, hit=0)


class Runesword(BaseBroadsword):

    def __init__(self):
        super().__init__('runesword', weight=40, damage=D.Dice.from_str('d6+1'), material=M.Iron, hit=0)


class ElvenBroadsword(BaseBroadsword):

    def __init__(self):
        super().__init__('elven broadsword', weight=70, damage=D.Dice.from_str('d6+1'), material=M.Wood, hit=0)
