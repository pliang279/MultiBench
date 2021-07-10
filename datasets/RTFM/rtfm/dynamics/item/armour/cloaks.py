# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseCloak(Armour):
    pass


class MummyWrapping(BaseCloak):

    def __init__(self):
        super().__init__('mummy wrapping', weight=3, armour_class=0, material=M.Cloth)


class OrcishCloak(BaseCloak):

    def __init__(self):
        super().__init__('orcish cloak', weight=10, armour_class=0, material=M.Cloth)


class DwarvishCloak(BaseCloak):

    def __init__(self):
        super().__init__('dwarvish cloak', weight=10, armour_class=0, material=M.Cloth)


class LeatherCloak(BaseCloak):

    def __init__(self):
        super().__init__('leather cloak', weight=15, armour_class=1, material=M.Leather)


class OilskinCloak(BaseCloak):

    def __init__(self):
        super().__init__('oilskin cloak', weight=10, armour_class=1, material=M.Cloth)
