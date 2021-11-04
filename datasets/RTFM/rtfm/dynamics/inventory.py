# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from rtfm.dynamics import item as I
from collections import defaultdict


class Inventory:

    slots = {
        I.BaseShirt,
        I.BaseSuit,
        I.BaseCloak,
        I.BaseHelm,
        I.BaseGlove,
        I.BaseShield,
        I.BaseBoot,
        I.Weapon,
        I.Amulet,
        I.Ring,
    }

    def __init__(self, mapping=None):
        mapping = mapping or {}
        self.equipped = {}
        for k, v in mapping.items():
            self.equip(k, v)

    def copy(self):
        return copy.deepcopy(self)

    def describe(self):
        items = [o.describe() for o in self.equipped_items]
        return '{}'.format('; '.join(items))

    def __repr__(self):
        return '[{}]'.format(self.describe())

    @property
    def equipped_items(self):
        keys = list(self.equipped.keys())
        keys.sort(key=lambda k: k.__name__)
        return [self.equipped[k] for k in keys]

    @property
    def weight(self):
        return sum(i.weight for i in self.equipped.values())

    @property
    def armour_class(self):
        return sum(i.armour_class for i in self.equipped.values() if isinstance(i, I.Armour))

    @property
    def hit(self):
        return sum(i.hit for i in self.equipped.values() if isinstance(i, I.Weapon))

    @property
    def damage(self):
        return sum(i.compute_damage() for i in self.equipped.values() if isinstance(i, I.Weapon))

    @property
    def elemental_damage(self):
        dmg = defaultdict(lambda: 0)
        for i in self.equipped.values():
            if isinstance(i, I.Weapon):
                for k, v in i.elemental_damage.items():
                    dmg[k] += v
        return dmg

    @property
    def elemental_armour_class(self):
        dmg = defaultdict(lambda: 0)
        for i in self.equipped.values():
            if isinstance(i, I.Armour):
                for k, v in i.elemental_armour_class.items():
                    dmg[k] += v
        return dmg

    def contains(self, item):
        return item in set(list(self.equipped.values()))

    def equip(self, slot, item):
        assert slot in self.slots
        assert isinstance(item, slot)
        self.equipped[slot] = item

    def auto_equip(self, item):
        for s in self.slots:
            if isinstance(item, s):
                self.equip(s, item)
                return
        raise Exception(
            'Cannot equip item {} because could not find appropriate slot.'.format(item))

    def get_slot(self, slot):
        assert slot in self.slots
        return self.equipped.get(slot, None)

    def unequip(self, slot):
        assert slot in self.slots
        item = self.get_slot(slot)
        del self.equipped[slot]
        return item
