# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseShield(Armour):
    pass


class SmallShield(BaseShield):

    def __init__(self):
        super().__init__('small shield', weight=30, armour_class=1, material=M.Wood)


class OrcishShield(BaseShield):

    def __init__(self):
        super().__init__('orcish shield', weight=50, armour_class=1, material=M.Iron)


class UrukHaiShield(BaseShield):

    def __init__(self):
        super().__init__('Uruk-hai shield', weight=50, armour_class=1, material=M.Iron)


class ElvenShield(BaseShield):

    def __init__(self):
        super().__init__('elven shield', weight=40, armour_class=2, material=M.Wood)


class DwarvishRoundshield(BaseShield):

    def __init__(self):
        super().__init__('dwarvish roundshield', weight=100, armour_class=2, material=M.Iron)


class LargeShield(BaseShield):

    def __init__(self):
        super().__init__('large shield', weight=100, armour_class=2, material=M.Iron)
