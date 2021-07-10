# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseDagger(Weapon):
    pass


class OrcishDagger(BaseDagger):

    def __init__(self):
        super().__init__('orcish dagger', weight=10, damage=D.Dice.from_str('d3'), material=M.Iron, hit=2)


class Dagger(BaseDagger):

    def __init__(self):
        super().__init__('dagger', weight=10, damage=D.Dice.from_str('d3'), material=M.Iron, hit=2)


class SilverDagger(BaseDagger):

    def __init__(self):
        super().__init__('silver dagger', weight=12, damage=D.Dice.from_str('d3'), material=M.Silver, hit=2)


class Athame(BaseDagger):

    def __init__(self):
        super().__init__('athame', weight=10, damage=D.Dice.from_str('d3'), material=M.Iron, hit=2)


class ElvenDagger(BaseDagger):

    def __init__(self):
        super().__init__('elven dagger', weight=10, damage=D.Dice.from_str('d3'), material=M.Wood, hit=2)
