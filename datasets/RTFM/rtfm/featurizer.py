# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import revtok
import random
from pprint import pprint
from rtfm.dynamics import monster as M, item as I, world_object as O, event as E


class Featurizer:

    def get_observation_space(self, task):
        raise NotImplementedError()

    def featurize(self, task):
        raise NotImplementedError()


class Concat(Featurizer, list):

    def get_observation_space(self, task):
        feat = {}
        for f in self:
            feat.update(f.get_observation_space(task))
        return feat

    def featurize(self, task):
        feat = {}
        for f in self:
            feat.update(f.featurize(task))
        return feat


class ValidMoves(Featurizer):

    def can_move_to(self, agent, pos, world):
        x, y = pos
        existing = world.get_objects_at_pos((x, y))
        can_inhabit = all([agent.can_inhabit_cell(o) for o in existing])
        return 0 <= x < world.width and 0 <= y <= world.height and can_inhabit

    def get_observation_space(self, task):
        return {'valid': (len(M.BaseMonster.valid_moves), )}

    def featurize(self, task):
        valid = set(M.BaseMonster.valid_moves)
        if task.agent is not None and task.agent.position is not None:
            x, y = task.agent.position
            if not self.can_move_to(task.agent, (x-1, y), task.world):
                valid.remove(E.Left)
            if not self.can_move_to(task.agent, (x+1, y), task.world):
                valid.remove(E.Right)
            if not self.can_move_to(task.agent, (x, y-1), task.world):
                valid.remove(E.Up)
            if not self.can_move_to(task.agent, (x, y+1), task.world):
                valid.remove(E.Down)
        return {'valid': torch.tensor([a in valid for a in M.BaseMonster.valid_moves], dtype=torch.float)}


class Position(Featurizer):

    def get_observation_space(self, task):
        return {'position': (2, )}

    def featurize(self, task):
        feat = [0, 0]
        valid = set(M.BaseMonster.valid_moves)
        if task.agent is not None and task.agent.position is not None:
            x, y = task.agent.position
            feat = [x, y]
        return {'position': torch.tensor(feat, dtype=torch.long)}


class RelativePosition(Featurizer):

    def get_observation_space(self, task):
        return {'rel_pos': (task.world.height, task.world.width, 2)}

    def featurize(self, task):
        x_offset = torch.Tensor(task.world.height, task.world.width).zero_()
        y_offset = torch.Tensor(task.world.height, task.world.width).zero_()
        if task.agent is not None and task.agent.position is not None:
            x, y = task.agent.position
            for i in range(task.world.width):
                x_offset[:, i] = i - x
            for i in range(task.world.height):
                y_offset[i, :] = i - y
        return {'rel_pos': torch.stack([x_offset/task.world.width, y_offset/task.world.height], dim=2)}


class WikiExtract(Featurizer):

    def get_observation_space(self, task):
        return {
            'wiki_extract': (task.max_wiki, ),
        }

    def featurize(self, task):
        return {'wiki_extract': task.get_wiki_extract()}


class Progress(Featurizer):

    def get_observation_space(self, task):
        return {'progress': (1, )}

    def featurize(self, task):
        return {'progress': torch.tensor([task.iter / task.max_iter], dtype=torch.float)}


class Terminal(Featurizer):

    def get_observation_space(self, task):
        return {}

    def clear(self):
        # for windows
        if os.name == 'nt':
            _ = os.system('cls')
            # for mac and linux(here, os.name is 'posix')
        else:
            _ = os.system('clear')

    def featurize(self, task):
        self.clear()
        print("\r")
        print(task.world.render(perspective=task.perspective))

        print('-' * 80)
        print('Wiki')
        print(task.get_wiki())
        print('Task:')
        print(task.get_task())
        print('Inventory:')
        print(task.get_inv())

        print('-' * 80)
        print('Last turn:')
        print('-' * 80)
        if task.history:
            for event in task.history[-1]:
                print(event)

        print('-' * 80)
        print('Monsters:')
        print('-' * 80)
        for m in task.world.monsters:
            print('{}: {}'.format(m.char, m))
            print(m.describe())
            print()

        print('-' * 80)
        print('Items:')
        print('-' * 80)
        for m in task.world.items:
            print('{}: {}'.format(m.char, m))
            print(m.describe())
            print()

        print()
        pprint(M.Player.keymap)
        return {}


class Symbol(Featurizer):

    class_list = [
        O.Empty,
        O.Unobservable,

        O.Wall,
        O.Door,

        M.HostileMonster,

        M.QueuedAgent,
    ]

    class_map = {c: i for i, c in enumerate(class_list)}

    def __init__(self):
        self.num_symbols = len(self.class_list)

    def get_observation_space(self, task):
        return {
            'symbol': (*task.world_shape, task.max_placement),
        }

    def featurize(self, task):
        mat = task.world.get_observation(
            perspective=task.perspective, max_placement=task.max_placement)
        smat = []
        for y in range(0, len(mat)):
            row = []
            for x in range(0, len(mat[0])):
                os = mat[y][x]
                classes = [self.class_map[o.__class__] for o in os]
                row.append(classes)
            smat.append(row)
        return {'symbol': torch.tensor(smat, dtype=torch.long)}


class Text(Featurizer):

    def __init__(self, max_cache=1e6):
        super().__init__()
        self._cache = {}
        self.max_cache = max_cache

    def get_observation_space(self, task):
        return {
            'name': (*task.world_shape, task.max_placement, task.max_name),
            'name_len': (*task.world_shape, task.max_placement),
            'inv': (task.max_inv, ),
            'inv_len': (1, ),
            'wiki': (task.max_wiki, ),
            'wiki_len': (1, ),
            'task': (task.max_task, ),
            'task_len': (1, ),
        }

    def featurize(self, task, eos='pad', pad='pad'):
        mat = task.world.get_observation(
            perspective=task.perspective, max_placement=task.max_placement)
        smat = []
        lmat = []
        for y in range(0, len(mat)):
            srow = []
            lrow = []

            for x in range(0, len(mat[0])):
                names = []
                lengths = []
                for o in mat[y][x]:
                    n, l = self.lookup_sentence(
                        o.describe(), task.vocab, max_len=task.max_name, eos=eos, pad=pad)
                    names.append(n)
                    lengths.append(l)
                srow.append(names)
                lrow.append(lengths)
            smat.append(srow)
            lmat.append(lrow)
        wiki, wiki_length = self.lookup_sentence(task.get_tokenized_wiki() if hasattr(
            task, 'get_tokenized_wiki') else task.get_wiki(), task.vocab, max_len=task.max_wiki, eos=eos, pad=pad)
        ins, ins_length = self.lookup_sentence(task.get_tokenized_task() if hasattr(
            task, 'get_tokenized_task') else task.get_task(), task.vocab, max_len=task.max_task, eos=eos, pad=pad)
        inv, inv_length = self.lookup_sentence(
            task.get_inv(), task.vocab, max_len=task.max_inv, eos=eos, pad=pad)
        ret = {
            'name': smat,
            'name_len': lmat,
            'inv': inv,
            'inv_len': [inv_length],
            'wiki': wiki,
            'wiki_len': [wiki_length],
            'task': ins,
            'task_len': [ins_length],
        }
        ret = {k: torch.tensor(v, dtype=torch.long) for k, v in ret.items()}
        return ret

    def lookup_sentence(self, sent, vocab, max_len=10, eos='pad', pad='pad'):
        if isinstance(sent, list):
            words = sent[:max_len-1] + [eos]
            length = len(words)
            if len(words) < max_len:
                words += [pad] * (max_len - len(words))
            return vocab.word2index([w.strip() for w in words]), length
        else:
            sent = sent.lower()
            key = sent, max_len
            if key not in self._cache:
                words = revtok.tokenize(sent)[:max_len-1] + [eos]
                length = len(words)
                if len(words) < max_len:
                    words += [pad] * (max_len - len(words))
                self._cache[key] = vocab.word2index(
                    [w.strip() for w in words]), length
                while len(self._cache) > self.max_cache:
                    keys = list(self._cache.keys())
                    del self._cache[random.choice(keys)]
            return self._cache[key]
