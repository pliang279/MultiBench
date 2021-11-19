# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import itertools
from rtfm.dynamics import monster as M, descriptor as D, item as I, element as types, inventory as V
from rtfm.tasks.room import RoomTask
from rtfm import featurizer as F, utils
from rtfm.tasks import groups_templates


ALL_TYPES = [types.Cold, types.Fire, types.Lightning, types.Poison]


def generate_all(all_monsters, all_groups, all_modifiers):
    # all monster assignments
    monster_groupings = set()
    monsters_per_group = len(all_monsters) // len(all_groups)
    for monsters in itertools.permutations(all_monsters, len(all_monsters)):
        groups = []
        for i in range(0, len(monsters), monsters_per_group):
            group = monsters[i:i+monsters_per_group]
            groups.append(frozenset(group))
        monster_groupings.add(frozenset(groups))
    monster_assignments = set()
    for groups in itertools.permutations(all_groups, len(all_groups)):
        for monster_grouping in monster_groupings:
            assignment = set()
            for g, mg in zip(groups, monster_grouping):
                assignment.add((g, tuple(sorted(list(mg)))))
            monster_assignments.add(frozenset(assignment))

    # all item assignments
    modifier_groupings = set()
    modifiers_per_element = len(all_modifiers) // len(ALL_TYPES)
    for modifiers in itertools.permutations(all_modifiers, len(all_modifiers)):
        groups = []
        for i in range(0, len(modifiers), modifiers_per_element):
            group = modifiers[i:i+modifiers_per_element]
            groups.append(frozenset(group))
        modifier_groupings.add(frozenset(groups))
    modifier_assignments = set()
    for elements in itertools.permutations(list(range(len(ALL_TYPES))), len(ALL_TYPES)):
        for modifier_grouping in modifier_groupings:
            assignment = []
            for e, mg in zip(elements, modifier_grouping):
                assignment.append((e, tuple(sorted(list(mg)))))
            modifier_assignments.add(frozenset(assignment))

    all_assignments = []
    for m in monster_assignments:
        for mm in modifier_assignments:
            all_assignments.append((m, mm))
    all_assignments.sort()

    random.Random(0).shuffle(all_assignments)

    n = len(all_assignments) // 2
    train = all_assignments[:n]
    dev = all_assignments[n:]
    return train, dev


class Groups(RoomTask):

    monsters = ['wolf', 'jaguar', 'panther', 'goblin',
                'bat', 'imp', 'shaman', 'ghost', 'zombie']
    groups = ['star alliance', 'order of the forest', 'rebel enclave']
    modifiers = [
        'grandmasters', 'blessed', 'shimmering', 'gleaming', 'fanatical', 'mysterious', 'soldiers', 'arcane'
    ]
    items = [
        'sword', 'axe', 'morningstar', 'polearm', 'knife', 'katana', 'cutlass', 'spear',
    ]
    config_index = 0
    default_max_iter = 1000

    class Monster(M.HostileMonster):
        char = '!'

        def __init__(self, element, name):
            # these will show up as "{name} dragon"
            super().__init__(
                name='{} {}'.format(element.describe(), name),
                aggression=0.6,
                constitution=10,
                strength=100,
                dexterity=1,
                intelligence=5,
                armour_class=1,
                speed=1,
            )
            self.monster_name = name
            self.element = element
            self.helmet = I.Helmet()
            for e in ALL_TYPES:
                if e != element:
                    self.helmet.add_elemental_armour_class(e, 100)
            self.inventory.auto_equip(self.helmet)
            self.inventory.auto_equip(I.Unarmed(hit=100, damage='100'))

    class Agent(M.QueuedAgent):

        def __init__(self):
            super().__init__(
                name='you',
                constitution=1,
                strength=1,
                dexterity=1,
                intelligence=5,
                armour_class=1,
                speed=2,
            )

    def __init__(self, room_shape=(10, 10), featurizer=F.Progress(), partially_observable=False, max_placement=2, max_name=8, max_inv=10, max_wiki=80, max_task=40, time_penalty=-0.02, shuffle_wiki=False):
        self.configs = generate_all(self.monsters, self.groups, self.modifiers)[
            self.config_index]
        # what group of enemies to target
        self.target_monster = None
        self.target_group = None
        self.distractor_monster = None
        self.distractor_item = None
        self.modifier_assignment = []
        self.group_assignment = []
        self._cache = {}
        super().__init__(room_shape, featurizer, partially_observable, self.default_max_iter,
                         max_placement, max_name, max_inv, max_wiki, max_task, time_penalty, shuffle_wiki=shuffle_wiki)

    def get_reward_finish_win(self):
        agent_dead = self.agent_is_dead()
        killed_enemy = not self.target_monster.is_alive(
        ) or not self.distractor_monster.is_alive()
        killed_correct_enemy = not self.target_monster.is_alive()
        finished = killed_enemy or self.out_of_turns() or agent_dead
        won = killed_correct_enemy and not agent_dead

        r = self.time_penalty
        if finished:
            if won:
                r = 1
            else:
                r = -1
        return r, finished, won

    def get_task(self):
        return 'defeat the {}'.format(self.target_group)

    def get_wiki(self):
        facts = []
        for element, modifiers in self.modifier_assignment:
            facts += ['{} beat {}.'.format(', '.join(modifiers),
                                           element.describe())]
        for group, monsters in self.group_assignment:
            facts += ['{} are {}.'.format(', '.join(monsters), group)]
        return ' '.join(facts)

    def get_wiki_extract(self):
        labels = []
        words = utils.tokenize(self.get_wiki())
        target_element = self.target_monster.element.describe()
        target_modifiers = set(
            [a for t, a in self.modifier_assignment if t == self.target_monster.element][0])
        target_monsters = set(
            [a for t, a in self.group_assignment if t == self.target_group][0])
        for w in words:
            l = 0
            if w == self.target_group:
                l = 1
            elif w == target_element:
                l = 2
            elif w in target_monsters:
                l = 3
            elif w in target_modifiers:
                l = 4
            labels.append(l)
        if len(labels) < self.max_wiki:
            labels += [255] * (self.max_wiki - len(labels))
        return torch.LongTensor(labels)

    def build_vocab(self):
        super().build_vocab()
        self.add_words(
            'cold fire poison lightning you are beat , . {  }'.split(' '))
        for n in Groups.monsters + Groups.modifiers + Groups.groups + Groups.items:
            self.add_words(n.split())
        for template in groups_templates.beat_utterance + groups_templates.group_utterance + groups_templates.task_utterance:
            words = utils.tokenize(template)
            self.add_words(words)

    def place_object(self, o):
        pos = self.world.get_random_placeable_location(tries=20)
        o.place(pos, self.world)
        return o

    def _reset(self):
        super()._reset()
        self._cache.clear()
        self.group_assignment.clear()
        self.modifier_assignment.clear()

        # sample dynamics
        sample_group, sample_mod = random.choice(self.configs)
        for group, monsters in sorted(list(sample_group)):
            self.group_assignment.append((group, monsters))
        for element, modifiers in sorted(list(sample_mod)):
            self.modifier_assignment.append((ALL_TYPES[element], modifiers))

        self.agent = self.place_object(self.Agent())

        self.target_group, target_monsters = random.choice(
            self.group_assignment)

        # choose a target element
        target_element, target_modifiers = random.choice(
            self.modifier_assignment)

        # choose a target monster
        self.target_monster = self.place_object(self.Monster(
            target_element, name=random.choice(target_monsters)))

        # create a target item
        good = self.place_object(I.Unarmed(hit=100, damage='1'))
        good.add_elemental_damage(target_element, dmg=50)
        good.name = '{} {}'.format(random.choice(
            target_modifiers), random.choice(self.items))
        good.char = 'y'

        # create a distractor item
        self.distractor_item = bad = self.place_object(
            I.Unarmed(hit=100, damage='1'))
        bad_element, bad_modifiers = random.choice(
            [m for m in self.modifier_assignment if m[0] != target_element])
        bad.add_elemental_damage(bad_element, dmg=50)
        bad.name = '{} {}'.format(random.choice(
            bad_modifiers), random.choice(self.items))
        bad.char = 'n'

        # create a distractor monster
        bad_group, bad_monsters = random.choice(
            [g for g in self.group_assignment if g[0] != self.target_group])
        self.distractor_monster = self.place_object(
            self.Monster(bad_element, name=random.choice(bad_monsters)))
        self.distractor_monster.char = '?'


class GroupsDev(Groups):
    config_index = 1


class GroupsStationary(Groups):

    class Monster(M.StationaryMonster):
        char = '!'

        def __init__(self, element, name):
            # these will show up as "{name} dragon"
            super().__init__(
                name='{} {}'.format(element.describe(), name),
                aggression=0.6,
                constitution=10,
                strength=100,
                dexterity=1,
                intelligence=5,
                armour_class=1,
                speed=1,
            )
            self.element = element
            self.helmet = I.Helmet()
            for e in ALL_TYPES:
                if e != element:
                    self.helmet.add_elemental_armour_class(e, 100)
            self.inventory.auto_equip(self.helmet)
            self.inventory.auto_equip(I.Unarmed(hit=100, damage='100'))


class GroupsStationaryDev(GroupsStationary):
    config_index = 1


class GroupsSimple(Groups):
    monsters = Groups.monsters[:3]
    modifiers = Groups.modifiers[:4]


class GroupsSimpleDev(GroupsSimple):
    config_index = 1


class GroupsSimpleStationary(GroupsStationary):
    monsters = Groups.monsters[:3]
    modifiers = Groups.modifiers[:4]


class GroupsSimpleStationaryDev(GroupsSimpleStationary):
    config_index = 1


class GroupsSimpleStationarySingleMonster(GroupsSimpleStationary):

    def _reset(self):
        super()._reset()
        self.world.remove_object(self.distractor_monster)


class GroupsSimpleStationarySingleMonsterDev(GroupsSimpleStationarySingleMonster):
    config_index = 1


class GroupsSimpleStationarySingleItem(GroupsSimpleStationary):

    def _reset(self):
        super()._reset()
        self.world.remove_object(self.distractor_item)


class GroupsSimpleStationarySingleItemDev(GroupsSimpleStationarySingleItem):
    config_index = 1


class NLGroups(Groups):

    use_cache = False

    def __init__(self, *args, **kwargs):
        self._cache = {}
        super().__init__(*args, **kwargs)

    def tokenize(self, sent):
        if sent not in self._cache:
            self._cache[sent] = utils.tokenize(sent)
        return self._cache[sent]

    def split_utterance(self, words, replace_from, replace_to):
        i = words.index(replace_from)
        return words[:i] + self.tokenize(replace_to) + words[i+1:]

    def get_beat_utterance(self, element, modifiers):
        t = random.choice(groups_templates.beat_utterance)
        t = self.tokenize(t)
        t = self.split_utterance(t, 'element', element)
        t = self.split_utterance(t, 'modifiers', ', '.join(modifiers))
        return t

    def get_group_utterance(self, group, monsters):
        t = random.choice(groups_templates.group_utterance)
        t = self.tokenize(t)
        t = self.split_utterance(t, 'group', group)
        t = self.split_utterance(t, 'monsters', ', '.join(monsters))
        return t

    def get_tokenized_task(self):
        if self.use_cache and 'task' in self._cache:
            t = self._cache['task']
        else:
            t = random.choice(groups_templates.task_utterance)
            t = self.tokenize(t)
            t = self.split_utterance(t, 'group', self.target_group)
            self._cache['task'] = t
        return t

    def get_task(self):
        return ' '.join(self.get_tokenized_task())

    def get_tokenized_wiki(self):
        if self.use_cache and 'wiki' in self._cache:
            cat = self._cache['wiki']
        else:
            facts = []
            for element, modifiers in self.modifier_assignment:
                facts.append(self.get_beat_utterance(
                    element.describe(), modifiers))
            for group, monsters in self.group_assignment:
                facts.append(self.get_group_utterance(group, monsters))
            random.shuffle(facts)
            cat = []
            for f in facts:
                cat += f
            self._cache['wiki'] = cat
        return cat

    def get_wiki(self):
        return ' '.join(self.get_tokenized_wiki())


class GroupsNL(NLGroups):
    use_cache = False
    pass


class GroupsNLDev(GroupsNL):
    use_cache = True
    config_index = 1
    default_max_iter = 100


class GroupsSimpleNL(NLGroups, GroupsSimple):
    pass


class GroupsSimpleNLDev(GroupsSimpleNL):
    config_index = 1


class GroupsStationaryNL(NLGroups, GroupsStationary):
    pass


class GroupsStationaryNLDev(GroupsStationaryNL):
    config_index = 1


class GroupsSimpleStationaryNL(NLGroups, GroupsSimpleStationary):
    pass
