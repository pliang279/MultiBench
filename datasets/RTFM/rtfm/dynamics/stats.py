# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy


class Stats:

    def __init__(self, constitution=1, strength=1, dexterity=1, intelligence=5, aggression=1., armour_class=1, speed=1):
        self.constitution = constitution
        self.strength = strength
        self.dexterity = dexterity
        self.armour_class = armour_class
        self.speed = speed
        self.intelligence = intelligence
        self.aggression = aggression

    def copy(self):
        return copy.deepcopy(self)
