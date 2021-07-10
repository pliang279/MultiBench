# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseHelm(Armour):
    pass


class Fedora(BaseHelm):

    def __init__(self):
        super().__init__('fedora', weight=3, armour_class=0, material=M.Cloth)


class DentedPot(BaseHelm):

    def __init__(self):
        super().__init__('dented pot', weight=10, armour_class=1, material=M.Iron)


class ElvenLeatherHelm(BaseHelm):

    def __init__(self):
        super().__init__('elven leather helm', weight=3, armour_class=1, material=M.Leather)


class Helmet(BaseHelm):

    def __init__(self):
        super().__init__('helmet', weight=30, armour_class=1, material=M.Iron)


class OrcishHelm(BaseHelm):

    def __init__(self):
        super().__init__('orcish helm', weight=30, armour_class=1, material=M.Iron)


class DwarvishIronHelm(BaseHelm):

    def __init__(self):
        super().__init__('dwarvish iron helm', weight=40, armour_class=2, material=M.Iron)
