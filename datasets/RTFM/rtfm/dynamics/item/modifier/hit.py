# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_modifier import BaseWeaponModifier, BasePrefixModifier


class BaseHitModifier(BaseWeaponModifier):
    level = 0

    @classmethod
    def apply_weapon(cls, weapon):
        weapon.hit += cls.level


class Sharp(BaseHitModifier, BasePrefixModifier):
    level = 1


class Fine(BaseHitModifier, BasePrefixModifier):
    level = 2


class Warrior(BaseHitModifier, BasePrefixModifier):
    level = 3


class Soldier(BaseHitModifier, BasePrefixModifier):
    level = 4

    @classmethod
    def describe(cls):
        return "soldier's"


class Knight(BaseHitModifier, BasePrefixModifier):
    level = 5

    @classmethod
    def describe(cls):
        return "knight's"


class Lord(BaseHitModifier, BasePrefixModifier):
    level = 6

    @classmethod
    def describe(cls):
        return "lord's"


class King(BaseHitModifier, BasePrefixModifier):
    level = 7

    @classmethod
    def describe(cls):
        return "king's"


class Master(BaseHitModifier, BasePrefixModifier):
    level = 8

    @classmethod
    def describe(cls):
        return "master's"


class Grandmaster(BaseHitModifier, BasePrefixModifier):
    level = 9

    @classmethod
    def describe(cls):
        return "grandmaster's"
