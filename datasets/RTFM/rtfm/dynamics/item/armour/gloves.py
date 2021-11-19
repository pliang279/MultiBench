# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_armour import Armour
from ... import material as M


class BaseGlove(Armour):
    pass


class LeatherGloves(BaseGlove):

    def __init__(self):
        super().__init__('leather gloves', weight=10, armour_class=1, material=M.Leather)
