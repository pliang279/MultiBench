# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_modifier import BaseArmourModifier, BasePrefixModifier


class BaseArmourClassModifier(BaseArmourModifier):
    level = 0

    @classmethod
    def apply_armour(cls, item):
        item.armour_class += cls.level


class Sturdy(BaseArmourClassModifier, BasePrefixModifier):
    level = 1


class Strong(BaseArmourClassModifier, BasePrefixModifier):
    level = 2


class Glorious(BaseArmourClassModifier, BasePrefixModifier):
    level = 3


class Blessed(BaseArmourClassModifier, BasePrefixModifier):
    level = 4


class Saintly(BaseArmourClassModifier, BasePrefixModifier):
    level = 5


class Holy(BaseArmourClassModifier, BasePrefixModifier):
    level = 6


class Godly(BaseArmourClassModifier, BasePrefixModifier):
    level = 7
