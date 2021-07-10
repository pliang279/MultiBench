# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Encumbrance:

    speed_multiplier = 1
    to_hit_modifier = 0

    @classmethod
    def get_encumbrance_state(cls, weight, capacity):
        if weight <= capacity:
            return Unencumbered
        elif weight <= 1.5 * capacity:
            return Burdened
        elif weight <= 2 * capacity:
            return Stressed
        elif weight <= 2.5 * capacity:
            return Strained
        elif weight <= 3 * capacity:
            return Overtaxed
        else:
            return Overloaded

    @classmethod
    def modify_speed(cls, speed):
        return speed * cls.speed_multiplier

    @classmethod
    def modify_hit(cls, to_hit):
        return to_hit + cls.to_hit_modifier

    @classmethod
    def describe(cls):
        return cls.__name__.lower()


class Unencumbered(Encumbrance):
    speed_multiplier = 1
    to_hit_modifier = 0


class Burdened(Encumbrance):
    speed_multiplier = 0.75
    to_hit_modifier = -1


class Stressed(Encumbrance):
    speed_multiplier = 0.5
    to_hit_modifier = -3


class Strained(Encumbrance):
    speed_multiplier = 0.25
    to_hit_modifier = -5


class Overtaxed(Encumbrance):
    speed_multiplier = 0.125
    to_hit_modifier = -7


class Overloaded(Encumbrance):
    speed_multiplier = 0
    to_hit_modifier = -9
