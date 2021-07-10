# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseClub(Weapon):
    pass


class Club(BaseClub):

    def __init__(self):
        super().__init__('club', weight=30, damage=D.Dice.from_str('d3'), material=M.Wood, hit=0)


class Aklys(BaseClub):

    def __init__(self):
        super().__init__('aklys', weight=15, damage=D.Dice.from_str('d3'), material=M.Iron, hit=0)
