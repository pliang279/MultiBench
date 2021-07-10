# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Modifier:

    @classmethod
    def describe(cls):
        keep = []
        last = 'X'
        for c in cls.__name__:
            if last.islower() and c.isupper():
                keep.append(' ')
            keep.append(c)
            last = c
        return ''.join(keep).lower()


class BaseArmourModifier(Modifier):

    @classmethod
    def apply_armour(cls, armour):
        raise NotImplementedError()


class BaseWeaponModifier(Modifier):

    @classmethod
    def apply_weapon(cls, weapon):
        raise NotImplementedError()


class BasePrefixModifier(Modifier):
    pass


class BaseSuffixModifier(Modifier):
    pass
