# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseSpear(Weapon):
    pass


class OrcishSpear(BaseSpear):

    def __init__(self):
        super().__init__('orcish spear', weight=30, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class Spear(BaseSpear):

    def __init__(self):
        super().__init__('spear', weight=30, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class SilverSpear(BaseSpear):

    def __init__(self):
        super().__init__('silver spear', weight=36, damage=D.Dice.from_str('d8'), material=M.Silver, hit=0)


class ElvenSpear(BaseSpear):

    def __init__(self):
        super().__init__('elven spear', weight=30, damage=D.Dice.from_str('d8'), material=M.Wood, hit=0)


class DwarvishSpear(BaseSpear):

    def __init__(self):
        super().__init__('dwarvish spear', weight=35, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class Javelin(BaseSpear):

    def __init__(self):
        super().__init__('javelin', weight=20, damage=D.Dice.from_str('d6'), material=M.Iron, hit=0)
