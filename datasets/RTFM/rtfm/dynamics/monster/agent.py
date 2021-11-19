# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import getch

from rtfm.dynamics import item as I, event as E
from rtfm.dynamics.monster.base import BaseMonster


class Agent(BaseMonster):
    char = '@'

    @classmethod
    def describe_class(cls):
        return "You"

    def target_is_portable(self, target):
        is_item = isinstance(target, I.BaseItem)
        return is_item


class Player(Agent):

    keymap = {
        'w': E.Up,
        's': E.Down,
        'a': E.Left,
        'd': E.Right,
        ' ': E.Stay,
    }

    def act(self, world, engine):
        user = None
        while user not in self.keymap:
            if user == 'h':
                print(self.keymap)

            user = getch.getche()
        engine.queue_event(self.keymap[user](self))


class QueuedAgent(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = []

    def act(self, world, engine):
        assert self.queue, 'No actions have been queued!'
        a = self.queue.pop(0)
        engine.queue_event(a)

    def queue_action(self, A):
        assert A in self.valid_moves, '{} is not a valid move in {}'.format(
            A, self.valid_moves)
        self.queue.append(A(actor=self))
