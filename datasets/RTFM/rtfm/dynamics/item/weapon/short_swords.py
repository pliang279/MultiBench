# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseShortSword(Weapon):
    pass


class OrcishShortSword(BaseShortSword):

    def __init__(self):
        super().__init__('orcish short sword', weight=30, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class ShortSword(BaseShortSword):

    def __init__(self):
        super().__init__('short sword', weight=30, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class DwarvishShortSword(BaseShortSword):

    def __init__(self):
        super().__init__('dwarvish short sword', weight=30, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class ElvenShortSword(BaseShortSword):

    def __init__(self):
        super().__init__('elven short sword', weight=30, damage=D.Dice.from_str('d8'), material=M.Wood, hit=0)
