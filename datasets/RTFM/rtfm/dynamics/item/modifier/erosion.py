# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_modifier import BaseWeaponModifier, Modifier, BaseArmourModifier


class BaseErosionModifier(BaseWeaponModifier, BaseArmourModifier):
    level = 0

    @classmethod
    def apply_weapon(cls, weapon):
        weapon.add_damage(-cls.level)

    @classmethod
    def apply_armour(cls, armour):
        armour.armour_class -= cls.level


class BaseRust(BaseErosionModifier):
    pass


class Rusty(BaseRust):
    level = 1


class VeryRusty(BaseRust):
    level = 2


class ThoroughlyRusty(BaseRust):
    level = 3


class BaseCorrosion(BaseErosionModifier):
    pass


class Corroded(BaseCorrosion):
    level = 1


class VeryCorroded(BaseCorrosion):
    level = 2


class ThoroughlyCorroded(BaseCorrosion):
    level = 3


class BaseBurn(BaseErosionModifier):
    pass


class Burnt(BaseBurn):
    level = 1


class VeryBurnt(BaseBurn):
    level = 2


class ThoroughlyBurnt(BaseBurn):
    level = 3


class BaseRot(BaseErosionModifier):
    pass


class Rotten(BaseRot):
    level = 1


class VeryRotten(BaseRot):
    level = 2


class ThoroughlyRotten(BaseRot):
    level = 3
