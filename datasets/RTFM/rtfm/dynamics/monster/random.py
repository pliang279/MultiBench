# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from rtfm.dynamics.monster.base import BaseMonster


class RandomMonster(BaseMonster):

    def act(self, world, engine):
        Move = random.choice(self.valid_moves)
        engine.queue_event(Move(self))
