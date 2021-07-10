# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BasePickAxe(Weapon):
    pass


class PickAxe(BasePickAxe):

    def __init__(self):
        super().__init__('pick-axe', weight=100, damage=D.Dice.from_str('d3'), material=M.Iron, hit=0)


class DwarvishMattock(BasePickAxe):

    def __init__(self):
        super().__init__('dwarvish mattock', weight=120, damage=D.Dice.from_str('d8+2d6'), material=M.Iron, hit=0)
