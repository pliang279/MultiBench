# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_modifier import BaseWeaponModifier, BasePrefixModifier, BaseSuffixModifier


class BaseDamageModifier(BaseWeaponModifier):
    level = 0

    @classmethod
    def apply_weapon(cls, weapon):
        weapon.add_damage(cls.level)


class Jagged(BaseDamageModifier, BasePrefixModifier):
    level = 1


class Deadly(BaseDamageModifier, BasePrefixModifier):
    level = 2


class Vicious(BaseDamageModifier, BasePrefixModifier):
    level = 3


class Brutal(BaseDamageModifier, BasePrefixModifier):
    level = 4


class Massive(BaseDamageModifier, BasePrefixModifier):
    level = 5


class Savage(BaseDamageModifier, BasePrefixModifier):
    level = 6


class Merciless(BaseDamageModifier, BasePrefixModifier):
    level = 7


class Ferocious(BaseDamageModifier, BasePrefixModifier):
    level = 8


class Cruel(BaseDamageModifier, BasePrefixModifier):
    level = 9






class Craftsmanship(BaseDamageModifier, BaseSuffixModifier):
    level = 1


class Quality(BaseDamageModifier, BaseSuffixModifier):
    level = 2


class Maiming(BaseDamageModifier, BaseSuffixModifier):
    level = 3


class Slaying(BaseDamageModifier, BaseSuffixModifier):
    level = 4


class Gore(BaseDamageModifier, BaseSuffixModifier):
    level = 5


class Carnage(BaseDamageModifier, BaseSuffixModifier):
    level = 6


class Slaughter(BaseDamageModifier, BaseSuffixModifier):
    level = 7


class Butchery(BaseDamageModifier, BaseSuffixModifier):
    level = 8


class Evisceration(BaseDamageModifier, BaseSuffixModifier):
    level = 9
