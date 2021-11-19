# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from rtfm.dynamics import event as E
from rtfm.dynamics.monster.base import BaseMonster


class HostileMonster(BaseMonster):

    def __init__(self, *args, aggression=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggression = aggression

    def target_is_attackable(self, target):
        is_base_monster = isinstance(target, BaseMonster)
        is_monster = isinstance(target, HostileMonster)
        return is_base_monster and not is_monster

    def act(self, world, engine):
        closer = []
        if random.random() < self.aggression:
            # find nearest agent that is observable
            agents = [
                a for a in world.agents if self.position_is_observable(a.position)]
            if agents:
                agents.sort(
                    key=lambda a: self.get_dist_to_position(a.position))
                target = agents[0]
                # find moves that would make us closer to agent
                x, y = self.position
                orig_dist = self.get_dist_to_position(target.position)
                if orig_dist == 0:
                    closer.append(E.Stay)
                else:
                    if x > 0 and target.get_dist_to_position((x-1, y)) < orig_dist:
                        closer.append(E.Left)
                    if x < world.width and target.get_dist_to_position((x+1, y)) < orig_dist:
                        closer.append(E.Right)
                    if y > 0 and target.get_dist_to_position((x, y-1)) < orig_dist:
                        closer.append(E.Up)
                    if y < world.height and target.get_dist_to_position((x, y+1)) < orig_dist:
                        closer.append(E.Down)
        if closer:
            Move = random.choice(closer)
            engine.queue_event(Move(self))
        else:
            # behave randomly
            super().act(world, engine)
