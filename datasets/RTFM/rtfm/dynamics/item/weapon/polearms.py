# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BasePolearm(Weapon):
    pass


class Partisan(BasePolearm):

    def __init__(self):
        super().__init__('partisan', weight=80, damage=D.Dice.from_str('d6+1'), material=M.Iron, hit=0)


class Fauchard(BasePolearm):

    def __init__(self):
        super().__init__('fauchard', weight=60, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class Glaive(BasePolearm):

    def __init__(self):
        super().__init__('glaive', weight=75, damage=D.Dice.from_str('d10'), material=M.Iron, hit=0)


class BecDeCorbin(BasePolearm):

    def __init__(self):
        super().__init__('bec-de-corbin', weight=100, damage=D.Dice.from_str('d6'), material=M.Iron, hit=0)


class Spetum(BasePolearm):

    def __init__(self):
        super().__init__('spetum', weight=50, damage=D.Dice.from_str('2d6'), material=M.Iron, hit=0)


class LucernHammer(BasePolearm):

    def __init__(self):
        super().__init__('lucern hammer', weight=150, damage=D.Dice.from_str('d6'), material=M.Iron, hit=0)


class Guisarme(BasePolearm):

    def __init__(self):
        super().__init__('guisarme', weight=80, damage=D.Dice.from_str('d8'), material=M.Iron, hit=0)


class Ranseur(BasePolearm):

    def __init__(self):
        super().__init__('ranseur', weight=50, damage=D.Dice.from_str('2d4'), material=M.Iron, hit=0)


class Voulge(BasePolearm):

    def __init__(self):
        super().__init__('voulge', weight=125, damage=D.Dice.from_str('2d4'), material=M.Iron, hit=0)


class BillGuisarme(BasePolearm):

    def __init__(self):
        super().__init__('bill-guisarme', weight=120, damage=D.Dice.from_str('d10'), material=M.Iron, hit=0)


class Bardiche(BasePolearm):

    def __init__(self):
        super().__init__('bardiche', weight=120, damage=D.Dice.from_str('3d4'), material=M.Iron, hit=0)


class Halberd(BasePolearm):

    def __init__(self):
        super().__init__('halberd', weight=150, damage=D.Dice.from_str('2d6'), material=M.Iron, hit=0)
