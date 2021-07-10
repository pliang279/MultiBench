# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class Unarmed(Weapon):

    def __init__(self, hit=0, damage='d2'):
        super().__init__('unarmed', weight=0, hit=hit, damage=D.Dice.from_str(damage), material=M.Flesh)
