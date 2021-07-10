# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseSuit(Armour):
    pass


class LeatherJacket(BaseSuit):

    def __init__(self):
        super().__init__('leather jacket', weight=30, armour_class=1, material=M.Leather)


class LeatherArmor(BaseSuit):

    def __init__(self):
        super().__init__('leather armor', weight=150, armour_class=2, material=M.Leather)


class OrcishRingMail(BaseSuit):

    def __init__(self):
        super().__init__('orcish ring mail', weight=250, armour_class=2, material=M.Iron)


class StuddedLeatherArmor(BaseSuit):

    def __init__(self):
        super().__init__('studded leather armor', weight=200, armour_class=3, material=M.Leather)


class RingMail(BaseSuit):

    def __init__(self):
        super().__init__('ring mail', weight=250, armour_class=3, material=M.Iron)


class ScaleMail(BaseSuit):

    def __init__(self):
        super().__init__('scale mail', weight=250, armour_class=4, material=M.Iron)


class OrcishChainMail(BaseSuit):

    def __init__(self):
        super().__init__('orcish chain mail', weight=300, armour_class=4, material=M.Iron)


class ChainMail(BaseSuit):

    def __init__(self):
        super().__init__('chain mail', weight=300, armour_class=5, material=M.Iron)


class ElvenMithrilCoat(BaseSuit):

    def __init__(self):
        super().__init__('elven mithril-coat', weight=150, armour_class=5, material=M.Mithril)


class SplintMail(BaseSuit):

    def __init__(self):
        super().__init__('splint mail', weight=400, armour_class=6, material=M.Iron)


class BandedMail(BaseSuit):

    def __init__(self):
        super().__init__('banded mail', weight=350, armour_class=6, material=M.Iron)


class DwarvishMithrilCoat(BaseSuit):

    def __init__(self):
        super().__init__('dwarvish mithril-coat', weight=150, armour_class=6, material=M.Mithril)


class BronzePlateMail(BaseSuit):

    def __init__(self):
        super().__init__('bronze plate mail', weight=450, armour_class=6, material=M.Copper)


class PlateMail(BaseSuit):

    def __init__(self):
        super().__init__('plate mail', weight=450, armour_class=7, material=M.Iron)


class CrystalPlateMail(BaseSuit):

    def __init__(self):
        super().__init__('crystal plate mail', weight=450, armour_class=7, material=M.Glass)
