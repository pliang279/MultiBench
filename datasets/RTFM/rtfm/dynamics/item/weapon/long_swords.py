# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseLongSword(Weapon):
    pass


class LongSword(BaseLongSword):

    def __init__(self):
        super().__init__('long sword', weight=40, damage=D.Dice.from_str('d12'), material=M.Iron, hit=0)


class Katana(BaseLongSword):

    def __init__(self):
        super().__init__('katana', weight=40, damage=D.Dice.from_str('d12'), material=M.Iron, hit=0)
