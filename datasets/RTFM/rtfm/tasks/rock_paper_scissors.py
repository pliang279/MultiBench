# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import string
import itertools
import numpy as np
from ..dynamics import monster as M, descriptor as D, item as I, element as types, inventory as V
from .room import RoomTask
from .. import featurizer as F


ALL_TYPES = [types.Cold, types.Fire, types.Lightning]


class RockPaperScissors(RoomTask):
    split_index = 0

    @classmethod
    def compute_labels(cls, limit=5):
        all_labels = string.ascii_lowercase[:limit]
        train = []
        dev = []
        perms = list(itertools.permutations(all_labels, 3))
        perms = sorted(list(perms))
        perms.sort()
        random.Random(0).shuffle(perms)
        n = len(perms) // 2
        train = perms[:n]
        dev = perms[n:]
        return train, dev

    class Dragon(M.HostileMonster):
        char = '!'

        def __init__(self, element, name):
            # these will show up as "{name} dragon"
            super().__init__(
                name='{} dragon'.format(name),
                aggression=0.6,
                constitution=1,
                strength=1,
                dexterity=1,
                intelligence=5,
                armour_class=1,
                speed=2,
            )
            self.element = element
            self.weapon = I.Unarmed(hit=100, damage='1')
            self.weapon.elemental_damage[self.element] += 100
            self.inventory.auto_equip(self.weapon)

    class Agent(M.QueuedAgent):

        def __init__(self):
            super().__init__(
                name='queuedagent',
                constitution=10,
                strength=1,
                dexterity=1,
                intelligence=5,
                armour_class=1,
                speed=1,
                inventory=V.Inventory({I.Weapon: I.Unarmed(hit=100, damage='100')})
            )

    def __init__(self, num_enemies=1, room_shape=(10, 10), featurizer=F.Progress(), partially_observable=False, max_iter=1000, max_placement=2, max_name=8, max_inv=10, max_wiki=80, max_task=40, time_penalty=-0.02, shuffle_wiki=False):
        self.labels = self.compute_labels()[self.split_index]
        self.TYPE_TO_MONSTER = {}
        self.TYPE_TO_ITEM = {}

        self.num_enemies = num_enemies
        self.enemies = []
        self.items = []
        self.type_index = 0
        super().__init__(room_shape, featurizer, partially_observable, max_iter, max_placement, max_name, max_inv, max_wiki, max_task, time_penalty, shuffle_wiki=shuffle_wiki)

    def set_types(self, labels):
        for i in range(len(labels)):
            self.TYPE_TO_MONSTER[ALL_TYPES[i]] = labels[i]
            self.TYPE_TO_ITEM[ALL_TYPES[i]] = labels[(i+1) % len(ALL_TYPES)]

    def get_reward_finish_win(self):
        agent_dead = self.agent_is_dead()
        killed_enemy = any([not e.is_alive() for e in self.enemies])

        finished = killed_enemy or self.out_of_turns() or agent_dead
        won = killed_enemy and not agent_dead

        r = self.time_penalty
        if finished:
            if won:
                r = 1
            else:
                r = -1
        return r, finished, won

    def get_task(self):
        return 'dragon.'

    def get_wiki(self):
        facts = []
        for el in ALL_TYPES:
            facts.append('{} beats {}.'.format(self.TYPE_TO_ITEM[el], self.TYPE_TO_MONSTER[el]))
        return ' '.join(facts)

    def build_vocab(self):
        super().build_vocab()
        self.add_words('; you beats dragon '.split(' '))
        self.add_words(list(string.ascii_lowercase))

    def place_object(self, o):
        pos = self.world.get_random_placeable_location(tries=20)
        o.place(pos, self.world)
        return o

    def _reset(self):
        super()._reset()
        self.enemies.clear()
        self.items.clear()

        self.agent = self.place_object(self.Agent())

        self.set_types(random.choice(self.labels))
        self.type_index = np.random.randint(0, len(ALL_TYPES))
        monster_type = ALL_TYPES[self.type_index]

        # create some dragons
        for i in range(self.num_enemies):
            self.enemies.append(self.place_object(self.Dragon(monster_type, self.TYPE_TO_MONSTER[monster_type])))

        # create some items
        for t in ALL_TYPES:
            o = I.Helmet()
            o.elemental_armour_class[t] += 100
            o.name = self.TYPE_TO_ITEM[t]
            o.char = o.name[-1]
            self.items.append(self.place_object(o))


class RockPaperScissorsDev(RockPaperScissors):
    split_index = 1


class RockPaperScissorsMed(RockPaperScissors):

    @classmethod
    def compute_labels(cls, limit=20):
        all_labels = string.ascii_lowercase[:limit]
        train = []
        dev = []
        train_vocab = set()
        for i in range(0, len(all_labels)-5, 2):
            a, b, c, d, e = all_labels[i:i+5]
            for trip in itertools.permutations([a, b, d]):
                train.append(trip)
                train_vocab |= set(trip)
            for trip in itertools.permutations([b, c, e]):
                dev.append(trip)
        dev = [(a, b, c) for a, b, c in dev if a in train_vocab and b in train_vocab and c in train_vocab]
        return train, dev


class RockPaperScissorsMedDev(RockPaperScissorsMed):
    split_index = 1


class RockPaperScissorsHard(RockPaperScissors):

    @classmethod
    def compute_labels(cls, limit=10):
        all_labels = string.ascii_lowercase[:limit]
        n = len(all_labels) // 2
        train = list(itertools.permutations(all_labels[:n], 3))
        dev = list(itertools.permutations(all_labels[n:], 3))
        return train, dev


class RockPaperScissorsHardDev(RockPaperScissorsHard):
    split_index = 1
