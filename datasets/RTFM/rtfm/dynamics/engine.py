# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from rtfm.dynamics import event as E
from rtfm import featurizer as F


class Engine:

    def __init__(self):
        self.event_queue = []

    def reset(self):
        self.event_queue.clear()

    def queue_event(self, event):
        self.event_queue.append(event)

    def queue_immediate_event(self, event):
        self.event_queue.insert(0, event)

    def run_turn(self, world):
        monsters = list(world.monsters)
        random.shuffle(monsters)
        monsters.sort(key=lambda m: m.speed, reverse=True)

        
        # for m in monsters:
        

        for agent in monsters:
            agent.act(world, self)

        executed = []
        while self.event_queue:
            event = self.event_queue.pop(0)
            agent = event.actor
            if not isinstance(event, E.Event):
                raise Exception('Unknown event {}'.format(event))
            if isinstance(event, E.Stay):
                agent.move_to_pos(agent.position, world, self)
            elif isinstance(event, E.Up):
                agent.move_to_pos(
                    (agent.position[0], agent.position[1]-1), world, self)
            elif isinstance(event, E.Down):
                agent.move_to_pos(
                    (agent.position[0], agent.position[1]+1), world, self)
            elif isinstance(event, E.Left):
                agent.move_to_pos(
                    (agent.position[0]-1, agent.position[1]), world, self)
            elif isinstance(event, E.Right):
                agent.move_to_pos(
                    (agent.position[0]+1, agent.position[1]), world, self)
            elif isinstance(event, E.Death):
                world.remove_object(agent)
                self.event_queue = [
                    e for e in self.event_queue if not e.contains_actor(agent)]
            executed.append(event)
        return executed

    def run(self, world, max_iter=10000, render=False):
        move_hist = []
        for turn in range(max_iter):
            executed = self.run_turn(world)
            move_hist.append(executed)
            if render:
                print(world.render())
                print(executed)
