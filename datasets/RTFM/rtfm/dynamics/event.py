# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Event:

    def __init__(self, actor):
        self.actor = actor

    def __repr__(self):
        return '{} performed {}'.format(self.actor, self.__class__.__name__)

    def contains_actor(self, actor):
        return actor is self.actor


class Movement(Event):
    pass


class Stay(Movement):
    pass


class Up(Movement):
    pass


class Down(Movement):
    pass


class Left(Movement):
    pass


class Right(Movement):
    pass


class Death(Event):
    pass


class Log(Event):
    pass


class Miss(Log):

    def __repr__(self):
        return '{} missed!'.format(self.actor)


class Damage(Log):

    def __init__(self, actor, victim, damage):
        super().__init__(actor)
        self.victim = victim
        self.damage = damage

    def __repr__(self):
        return '{} did {} damage to {}!'.format(self.actor, self.damage, self.victim)


class PickedUp(Log):

    def __init__(self, actor, item):
        super().__init__(actor)
        self.item = item

    def __repr__(self):
        return '{} picked up {}'.format(self.actor, self.item)
