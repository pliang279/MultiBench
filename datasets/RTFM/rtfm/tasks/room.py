# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from rtfm.tasks.task import Task
from rtfm.dynamics import world as W, monster as M
from rtfm import featurizer as F


class RoomTask(Task):
    # this environment is a base and not directly importable!

    def __init__(self, room_shape=(10, 10), featurizer=F.Progress(), partially_observable=False, max_iter=1000, max_placement=2, max_name=2, max_inv=10, max_wiki=80, max_task=40, time_penalty=-0.02, shuffle_wiki=False):
        self.world_shape = room_shape
        super().__init__(featurizer, partially_observable, max_iter, max_placement,
                         max_name, max_inv, max_wiki, max_task, time_penalty, shuffle_wiki=shuffle_wiki)
        self.observation_space = self.featurizer.get_observation_space(self)

    def build_vocab(self):
        super().build_vocab()
        self.add_words('move around welcome to rtfm .'.split())

    def get_task(self):
        return 'Move around.'

    def get_wiki(self):
        return 'Welcome to MiniHack.'

    def _reset(self):
        super()._reset()
        r = W.Room(*self.world_shape)
        r.place((0, 0), self.world)
