# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_modifier import *
from ... import element as element_types


class BaseElementalModifier(Modifier):
    elements = []
    level = 0


class BaseCold(BaseElementalModifier, element_types.Cold):
    elements = [element_types.Cold]


class BaseFire(BaseElementalModifier, element_types.Fire):
    elements = [element_types.Fire]


class BaseLightning(BaseElementalModifier, element_types.Lightning):
    elements = [element_types.Lightning]


class BasePoison(BaseElementalModifier, element_types.Poison):
    elements = [element_types.Poison]


class BaseAllElement(BaseElementalModifier):
    elements = [element_types.Fire, element_types.Cold, element_types.Poison, element_types.Lightning]


class BaseElementalWeaponModifier(BaseElementalModifier, BaseWeaponModifier):

    @classmethod
    def apply_weapon(cls, weapon):
        for e in cls.elements:
            weapon.add_elemental_damage(e, cls.level)


class BaseElementalArmourModifier(BaseElementalModifier, BaseArmourModifier):

    @classmethod
    def apply_armour(cls, armour):
        for e in cls.elements:
            armour.add_elemental_armour_class(e, cls.level)




class Snowy(BaseCold, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 1


class Shivering(BaseCold, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 2


class Boreal(BaseCold, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 3


class Hibernal(BaseCold, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 4


class Azure(BaseCold, BaseElementalArmourModifier, BasePrefixModifier):
    level = 1


class Lapis(BaseCold, BaseElementalArmourModifier, BasePrefixModifier):
    level = 2


class Cobalt(BaseCold, BaseElementalArmourModifier, BasePrefixModifier):
    level = 3


class Sapphire(BaseCold, BaseElementalArmourModifier, BasePrefixModifier):
    level = 4


class Frost(BaseCold, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 1


class Icicle(BaseCold, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 2


class Glacier(BaseCold, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 3


class Winter(BaseCold, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 4






class Fiery(BaseFire, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 1


class Smoldering(BaseFire, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 2


class Smoking(BaseFire, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 3


class Flaming(BaseFire, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 4


class Condensing(BaseFire, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 5


class Crimson(BaseFire, BaseElementalArmourModifier, BasePrefixModifier):
    level = 1


class Russet(BaseFire, BaseElementalArmourModifier, BasePrefixModifier):
    level = 2


class Garnet(BaseFire, BaseElementalArmourModifier, BasePrefixModifier):
    level = 3


class Ruby(BaseFire, BaseElementalArmourModifier, BasePrefixModifier):
    level = 4


class Flame(BaseFire, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 1


class Fire(BaseFire, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 2


class Burning(BaseFire, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 3


class Incineration(BaseFire, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 4





class Static(BaseLightning, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 1


class Glowing(BaseLightning, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 2


class Buzzing(BaseLightning, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 3


class Arcing(BaseLightning, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 4


class Shocking(BaseLightning, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 5


class Tangerine(BaseLightning, BaseElementalArmourModifier, BasePrefixModifier):
    level = 1


class Ocher(BaseLightning, BaseElementalArmourModifier, BasePrefixModifier):
    level = 2


class Coral(BaseLightning, BaseElementalArmourModifier, BasePrefixModifier):
    level = 3


class Amber(BaseLightning, BaseElementalArmourModifier, BasePrefixModifier):
    level = 4


class Shock(BaseLightning, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 1


class Lightning(BaseLightning, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 2


class Thunder(BaseLightning, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 3


class Storms(BaseLightning, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 4





class Septic(BasePoison, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 1


class Foul(BasePoison, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 2


class Corrosive(BasePoison, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 3


class Toxic(BasePoison, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 4


class Pestilent(BasePoison, BaseElementalWeaponModifier, BasePrefixModifier):
    level = 5


class Beryl(BasePoison, BaseElementalArmourModifier, BasePrefixModifier):
    level = 1


class Viridian(BasePoison, BaseElementalArmourModifier, BasePrefixModifier):
    level = 2


class Jade(BasePoison, BaseElementalArmourModifier, BasePrefixModifier):
    level = 3


class Emerald(BasePoison, BaseElementalArmourModifier, BasePrefixModifier):
    level = 4


class Blight(BasePoison, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 1


class Venom(BasePoison, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 2


class Pestilence(BasePoison, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 3


class Anthrax(BasePoison, BaseElementalWeaponModifier, BaseSuffixModifier):
    level = 4






class Shimmering(BaseAllElement, BaseElementalArmourModifier, BasePrefixModifier):
    level = 1


class Rainbow(BaseAllElement, BaseElementalArmourModifier, BasePrefixModifier):
    level = 2


class Scintillating(BaseAllElement, BaseElementalArmourModifier, BasePrefixModifier):
    level = 3


class Prismatic(BaseAllElement, BaseElementalArmourModifier, BasePrefixModifier):
    level = 4


class Chromatic(BaseAllElement, BaseElementalArmourModifier, BasePrefixModifier):
    level = 5


class Warding(BaseAllElement, BaseElementalArmourModifier, BaseSuffixModifier):
    level = 1


class Sentinel(BaseAllElement, BaseElementalArmourModifier, BaseSuffixModifier):
    level = 2


class Guarding(BaseAllElement, BaseElementalArmourModifier, BaseSuffixModifier):
    level = 3


class Negation(BaseAllElement, BaseElementalArmourModifier, BaseSuffixModifier):
    level = 4
