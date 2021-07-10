# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseBow(Weapon):
    pass


class OrcishBow(BaseBow):

    def __init__(self):
        super().__init__('orcish bow', weight=30, damage=D.Dice.from_str('d2'), material=M.Wood, hit=0)


class OrcishArrow(BaseBow):

    def __init__(self):
        super().__init__('orcish arrow', weight=1, damage=D.Dice.from_str('d6'), material=M.Iron, hit=0)


class Bow(BaseBow):

    def __init__(self):
        super().__init__('bow', weight=30, damage=D.Dice.from_str('d2'), material=M.Wood, hit=0)


class Arrow(BaseBow):

    def __init__(self):
        super().__init__('arrow', weight=1, damage=D.Dice.from_str('d6'), material=M.Iron, hit=0)


class ElvenBow(BaseBow):

    def __init__(self):
        super().__init__('elven bow', weight=30, damage=D.Dice.from_str('d2'), material=M.Wood, hit=0)


class ElvenArrow(BaseBow):

    def __init__(self):
        super().__init__('elven arrow', weight=1, damage=D.Dice.from_str('d6'), material=M.Wood, hit=0)


class Yumi(BaseBow):

    def __init__(self):
        super().__init__('yumi', weight=30, damage=D.Dice.from_str('d2'), material=M.Wood, hit=0)


class Ya(BaseBow):

    def __init__(self):
        super().__init__('ya', weight=1, damage=D.Dice.from_str('d7'), material=M.Metal, hit=0)


class SilverArrow(BaseBow):

    def __init__(self):
        super().__init__('silver arrow', weight=1, damage=D.Dice.from_str('d6'), material=M.Silver, hit=0)
