# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections import defaultdict, deque

from rtfm.dynamics import world_object as O, event as E, item as I, stats as S, inventory as V, encumbrance, dice as D


class BaseMonster(O.WorldObject):

    char = 'm'
    valid_moves = [E.Stay, E.Up, E.Down, E.Left, E.Right]
    descriptors = {
        'hit_points': ['near death', 'wounded', 'hurt', 'healthy'],
        'damage': ['weak', 'fledgling', 'strong', 'hulking'],
        'armour_class': ['naked', 'covered', 'armored', 'well protected'],
        'hit': ['uncoordinated', 'coordinated', 'accurate', 'precise'],
    }

    # note that the monster's armour class is negative by convention... so we flip it here
    armour_class_map = [3] * 4 + [2] * (6-4) + [1] * (8-6) + [0] * (
        15-8) + [-1, -2, -3, -4, -5] + [-6] * (22-20) + [-7] * (24-22) + [-8]
    armour_class_map = [-1 * x for x in armour_class_map]
    strength_hit_map = [-2] * 4 + [-2] * \
        (6-4) + [-1] * (8-6) + [0] * (17-8) + [1] * (19-17) + [3] * (26-19)
    strength_dmg_map = [-1] * 6 + [0] * \
        (16-6) + [1] * (18-16) + [2] + [6] * (25-19)
    dexterity_hit_map = [-3] * 4 + [-2] * (6-4) + [-1] * (8-6) + [0] * (15-8) + [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    combat_dice = D.SingleDice(20)

    def __init__(self, name=None, stats=None, inventory=None, **kwargs):
        super().__init__(name or self.__class__.__name__)
        self.stats = stats or S.Stats(**kwargs)
        self.inventory = inventory or V.Inventory()
        self.hit_points = self.max_hit_points

    def __repr__(self):
        return '{}({}, hp={}/{}, dmg={}, ac={})'.format(self.__class__.__name__, self.describe(), self.hit_points, self.max_hit_points, self.damage, self.armour_class)

    def describe(self):
        return self.name

    @property
    def sight(self):
        return self.stats.intelligence

    @property
    def weight(self):
        return self.inventory.weight

    @property
    def max_hit_points(self):
        return 10 + self.stats.constitution

    @property
    def carrying_capacity(self):
        return 25 * (self.stats.strength + self.stats.constitution) + 50

    @property
    def encumbrance(self):
        return encumbrance.Encumbrance.get_encumbrance_state(self.weight, self.carrying_capacity)

    @property
    def speed(self):
        return self.encumbrance.modify_speed(self.stats.speed)

    @property
    def natural_armour_class(self):
        if self.stats.dexterity < 0:
            return self.armour_class_map[0]
        elif self.stats.dexterity >= len(self.armour_class_map):
            return self.armour_class_map[-1]
        else:
            return self.armour_class_map[self.stats.dexterity]

    @property
    def natural_hit(self):
        if self.stats.strength < 0:
            from_strength = self.strength_hit_map[0]
        elif self.stats.strength >= len(self.strength_hit_map):
            from_strength = self.strength_hit_map[-1]
        else:
            from_strength = self.strength_hit_map[self.stats.strength]
        if self.stats.dexterity < 0:
            from_dexterity = self.dexterity_hit_map[0]
        elif self.stats.dexterity >= len(self.dexterity_hit_map):
            from_dexterity = self.dexterity_hit_map[-1]
        else:
            from_dexterity = self.dexterity_hit_map[self.stats.dexterity]
        return from_strength + from_dexterity

    @property
    def natural_damage(self):
        if self.stats.strength < 0:
            return self.strength_dmg_map[0]
        elif self.stats.strength >= len(self.strength_dmg_map):
            return self.strength_dmg_map[-1]
        else:
            return self.strength_dmg_map[self.stats.strength]

    @property
    def armour_class(self):
        return self.natural_armour_class + self.inventory.armour_class

    @property
    def damage(self):
        return self.natural_damage + self.inventory.damage

    @property
    def elemental_damage(self):
        return self.inventory.elemental_damage

    @property
    def elemental_armour_class(self):
        return self.inventory.elemental_armour_class

    @property
    def hit(self):
        return self.encumbrance.modify_hit(self.natural_hit + self.inventory.hit)

    @classmethod
    def calculate_to_hit(cls, attacker, victim):
        # attacker must roll < this to hit me
        return 10 - victim.armour_class + attacker.hit

    @classmethod
    def calculate_attack_success(cls, attacker, victim, engine=None):
        to_hit = cls.calculate_to_hit(attacker, victim)
        if to_hit <= 1:
            return False
        if to_hit >= 20:
            return True
        roll = cls.combat_dice.roll()
        if engine is not None:
            engine.queue_immediate_event(E.Log('{} rolled a combat roll of {} for {} to_hit of {} ({}% hit rate)'.format(
                attacker, roll, to_hit, victim, to_hit/20*100)))
        return roll < to_hit

    @classmethod
    def calculate_damage(cls, attacker, victim, engine=None):
        dmg = max(1, attacker.damage - victim.armour_class)
        for k, v in attacker.elemental_damage.items():
            v = max(0, v - victim.elemental_armour_class[k])
            dmg += v
        if engine is not None:
            engine.queue_immediate_event(E.Log('{} attacked with {} damage to {} armour class of {}'.format(
                attacker, attacker.damage, victim, victim.armour_class)))
        return dmg

    def attack(self, another, world, engine, override_success=False):
        success = override_success or self.calculate_attack_success(
            self, another, engine=engine)
        if success:
            damage = self.calculate_damage(self, another, engine=engine)
            another.hit_points -= damage
            if another.hit_points <= 0:
                engine.queue_immediate_event(E.Death(another))
            engine.queue_immediate_event(E.Damage(self, another, damage))
        else:
            engine.queue_immediate_event(E.Miss(self))

    def pickup(self, another, world, engine):
        world.remove_object(another)
        self.inventory.auto_equip(another)
        engine.queue_immediate_event(E.PickedUp(self, another))

    def target_is_attackable(self, target):
        is_monster = isinstance(target, BaseMonster)
        is_different = not isinstance(target, self.__class__)
        return is_monster and is_different

    def target_is_portable(self, target):
        # monsters can't pick up things
        return False

    def position_is_observable(self, position):
        return self.get_dist_to_position(position) <= self.sight

    def is_alive(self):
        return self.hit_points > 0

    def place(self, position, world):
        return super().place(position, world)

    def can_inhabit_cell(self, cell):
        return isinstance(cell, O.Empty) or (isinstance(cell, O.Door) and cell.open) or isinstance(cell, BaseMonster) or isinstance(cell, I.BaseItem)

    def can_inhabit_pos(self, pos, world):
        x, y = pos
        return 0 <= x < world.width and 0 <= y <= world.height and all([self.can_inhabit_cell(o) for o in world.get_objects_at_pos((x, y))])

    def get_valid_moves(self, world):
        valid = set(self.valid_moves)
        x, y = self.position
        if not self.can_inhabit_pos((x-1, y), world):
            valid.remove(E.Left)
        if not self.can_inhabit_pos((x+1, y), world):
            valid.remove(E.Right)
        if not self.can_inhabit_pos((x, y-1), world):
            valid.remove(E.Up)
        if not self.can_inhabit_pos((x, y+1), world):
            valid.remove(E.Down)
        return valid

    def act(self, world, engine):
        Move = random.choice(list(self.get_valid_moves(world)))
        engine.queue_event(Move(self))

    def move_to_pos(self, pos, world, engine):
        if self.can_inhabit_pos(pos, world):
            world.remove_object(self)
            self.place(pos, world)
            world.resolve_collision(self.position, engine)
