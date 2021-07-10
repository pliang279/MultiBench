# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_weapon import Weapon
from ... import dice as D, material as M


class BaseUnicornHorn(Weapon):
    pass


class UnicornHorn(BaseUnicornHorn):

    def __init__(self):
        super().__init__('unicorn horn', weight=20, damage=D.Dice.from_str('d12'), material=M.Bone, hit=0)
