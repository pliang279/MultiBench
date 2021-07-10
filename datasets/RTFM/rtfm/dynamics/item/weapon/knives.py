# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseKnive(Weapon):
    pass


class WormTooth(BaseKnive):

    def __init__(self):
        super().__init__('worm tooth', weight=20, damage=D.Dice.from_str('d2'), material=M.Undefined, hit=1)


class Knife(BaseKnive):

    def __init__(self):
        super().__init__('knife', weight=5, damage=D.Dice.from_str('d2'), material=M.Iron, hit=1)


class Stiletto(BaseKnive):

    def __init__(self):
        super().__init__('stiletto', weight=5, damage=D.Dice.from_str('d2'), material=M.Iron, hit=1)


class Scalpel(BaseKnive):

    def __init__(self):
        super().__init__('scalpel', weight=5, damage=D.Dice.from_str('d3'), material=M.Metal, hit=1)


class Crysknife(BaseKnive):

    def __init__(self):
        super().__init__('crysknife', weight=20, damage=D.Dice.from_str('d10'), material=M.Mineral, hit=1)
