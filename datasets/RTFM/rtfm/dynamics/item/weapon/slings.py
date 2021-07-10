# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseSling(Weapon):
    pass


class Sling(BaseSling):

    def __init__(self):
        super().__init__('sling', weight=3, damage=D.Dice.from_str('d2'), material=M.Leather, hit=0)


class Flintstone(BaseSling):

    def __init__(self):
        super().__init__('flintstone', weight=10, damage=D.Dice.from_str('d6'), material=M.Mineral, hit=0)
