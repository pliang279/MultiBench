# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseTwoHandedSword(Weapon):
    pass


class TwoHandedSword(BaseTwoHandedSword):

    def __init__(self):
        super().__init__('two-handed sword', weight=150, damage=D.Dice.from_str('3d6'), material=M.Iron, hit=0)


class Tsurugi(BaseTwoHandedSword):

    def __init__(self):
        super().__init__('tsurugi', weight=60, damage=D.Dice.from_str('d8+2d6'), material=M.Metal, hit=0)
