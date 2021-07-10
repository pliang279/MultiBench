# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import revtok
import copy


class WorldObject:
    """
    An object in the world
    """

    char = '?'

    def __init__(self, name):
        self.name = name
        self.position = None

    def __repr__(self):
        """
        Human readable description of the object
        """
        s = self.name
        if self.position is not None:
            s += repr(self.position)
        return s

    def copy(self):
        return copy.deepcopy(self)

    def describe(self):
        return self.name

    @classmethod
    def describe_class(cls):
        return cls.__name__

    def tokenized_name(self):
        return revtok.tokenize(self.name)

    def tokenized_description(self):
        return revtok.tokenize(self.describe()) if hasattr(self, 'describe') else self.tokenized_name()

    def render(self):
        """
        Draws a single character display
        """
        return self.char

    def place(self, position, w):
        w.place_object_at_pos(self, position)

    def get_path_to(self, another, world, ignore=None):
        queue = [(self.position, [])]  # start point, empty path
        visited = set()
        while len(queue) > 0:
            node, path = queue.pop(0)
            path.append(node)
            visited.add(node)

            if node == another.position:
                return path

            adj_nodes = world.get_neighbours(node, ignore=ignore)
            for adj_node in adj_nodes:
                if adj_node not in visited and world.contains_pos(adj_node):
                    queue.append((adj_node, path[:]))
        return None  # no path found

    def get_dist_to_position(self, position):
        sum_of_squares = 0
        for ai, bi in zip(self.position, position):
            sum_of_squares += (ai-bi)**2
        return np.sqrt(sum_of_squares)


class Structure(WorldObject):
    pass


class Wall(Structure):

    char = u"\u2588"

    def __init__(self, name='wall'):
        super().__init__(name)


class Door(Structure):

    def __init__(self, prefix='', open=False):
        super().__init__(prefix + ' door' if prefix else 'door')
        self.open = open
        self.char = '.' if self.open else '+'

    def link(self, another, world):
        path = self.get_path_to(another, world, ignore={Wall})
        if path is None:
            raise Exception('Path link error from {} to {}\nWorld\n{}'.format(self.position, another.position, world.render(' ')))
        path = path[1:-1]
        path_nodes = set(path)
        for x, y in path:
            # put some walls
            candidates = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            for pos in candidates:
                xc, yc = pos
                if pos not in path_nodes and 0 <= xc and 0 <= yc:
                    if world.is_empty_at(pos):
                        Wall().place(pos, world)


class Unobservable(WorldObject):
    char = '*'

    def __init__(self, name='fog'):
        super().__init__(name)


class Empty(WorldObject):
    char = ' '

    def __init__(self, name='empty'):
        super().__init__(name)
