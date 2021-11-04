# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
from collections import defaultdict
from itertools import combinations

from rtfm.dynamics import world_object as O, monster as M, item as I


class World:
    """
    A canvas for the grid world
    """

    def __init__(self, areas=None, map=None):
        self.map = map or defaultdict(lambda: set())
        self.areas = areas or set()
        self.monsters = set()
        self.agents = set()
        self.items = set()
        self.structure = set()
        self.xmin = self.ymin = self.xmax = self.ymax = 0
        self.recompute_extrema()

        self.EMPTY = O.Empty()
        self.UNOBSERVABLE = O.Unobservable()

    @property
    def objects(self):
        return sorted(list(self.monsters.union(self.items).union(self.structure)), key=lambda c: repr(c))

    @property
    def object_classes(self):
        return sorted(list({o.__class__ for o in self.objects}), key=lambda c: c.__name__)

    @property
    def height(self):
        return self.ymax + 1

    @property
    def width(self):
        return self.xmax + 1

    def reset(self):
        self.map.clear()
        self.areas.clear()
        self.monsters.clear()
        self.items.clear()
        self.agents.clear()
        self.structure.clear()
        self.recompute_extrema()

    def is_empty_at(self, position):
        return not self.map[position]

    def place_object_at_pos(self, obj, position):
        x, y = position
        assert x >= 0
        assert y >= 0
        assert obj.position is None, 'Object "{}" has already been placed at {}'.format(
            self, obj.position)

        self.map[position].add(obj)
        obj.position = position

        self.xmin = min(self.xmin, x)
        self.ymin = min(self.ymin, y)
        self.xmax = max(self.xmax, x)
        self.ymax = max(self.ymax, y)

        if isinstance(obj, M.BaseMonster):
            self.monsters.add(obj)
        if isinstance(obj, M.Agent):
            self.agents.add(obj)
        if isinstance(obj, I.BaseItem):
            self.items.add(obj)
        if isinstance(obj, O.Structure):
            self.structure.add(obj)

    def contains_obj(self, o):
        if o.position is None:
            return False
        return o in self.get_objects_at_pos(o.position)

    def get_objects_at_pos(self, position, perspective=None):
        if perspective is None:
            return self.map[position] or {self.EMPTY}
        else:
            if perspective.position is None or perspective.position_is_observable(position):
                return self.map[position] or {self.EMPTY}
            else:
                return {self.UNOBSERVABLE}

    def remove_object(self, obj):
        self.map[obj.position].remove(obj)
        if not self.map[obj.position]:
            del self.map[obj.position]
        obj.position = None
        if isinstance(obj, M.BaseMonster):
            self.monsters.remove(obj)
        if isinstance(obj, M.Agent):
            self.agents.remove(obj)
        if isinstance(obj, I.BaseItem):
            self.items.remove(obj)
        if isinstance(obj, O.Structure):
            self.structure.remove(obj)

    def remove_objects_at_pos(self, position):
        to_remove = list(self.map[position])
        for o in to_remove:
            self.remove_object(o)

    @classmethod
    def seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)

    def recompute_extrema(self):
        xmin = ymin = np.inf
        xmax = ymax = -1
        for (x, y), obj in self.map.items():
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def contains_pos(self, pos):
        x, y = pos
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def get_observation(self, perspective=None, max_placement=None):
        mat = []
        for y in range(0, self.height):
            row = []
            for x in range(0, self.width):
                os = [o for o in self.get_objects_at_pos(
                    (x, y), perspective=perspective)]
                os.sort(key=lambda o: getattr(o, 'speed', 0), reverse=True)
                if max_placement is not None:
                    os = os[:max_placement]
                    os += [self.EMPTY] * (max_placement - len(os))
                row.append(os)
            mat.append(row)
        return mat

    def render(self, empty_space=' ', perspective=None):
        mat = self.get_observation(perspective=perspective, max_placement=1)
        smat = []
        for y in range(0, self.height):
            row = ''
            for x in range(0, self.width):
                o = mat[y][x][0]
                row += o.render() if o is not self.EMPTY else empty_space
            smat.append(row)
        return '\n'.join(smat)

    def get_neighbours(self, pos, ignore=None):
        x, y = pos
        candidates = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        neighbours = []
        ignore = ignore or set()
        for x, y in candidates:
            if 0 <= x < self.width and 0 <= y < self.height:
                # is valid
                classes = [
                    o.__class__ for o in self.get_objects_at_pos((x, y))]
                if all([c not in ignore for c in classes]):
                    neighbours.append((x, y))
        return neighbours

    def get_random_placeable_location(self, tries=10):
        pos = None
        for i in range(tries):
            x = random.randint(1, self.width-2)
            y = random.randint(1, self.height-2)
            pos = x, y
            if self.is_empty_at(pos):
                break
        if not self.is_empty_at(pos):
            return None  # could not find placeable location
        return pos

    def resolve_collision(self, position, engine):
        objs = self.get_objects_at_pos(position)
        if len(objs) > 1:
            # there is collision here
            monsters, items = [], []
            for o in objs:
                if isinstance(o, M.BaseMonster):
                    monsters.append(o)
                if isinstance(o, I.BaseItem):
                    items.append(o)
            monsters.sort(key=lambda m: m.speed, reverse=True)
            for m in monsters:
                if items and m.target_is_portable(items[0]):
                    m.pickup(items.pop(0), self, engine)
            for a, b in combinations(monsters, 2):
                if a.is_alive() and b.is_alive() and a.target_is_attackable(b):
                    a.attack(b, self, engine)
                if a.is_alive() and b.is_alive() and b.target_is_attackable(a):
                    b.attack(a, self, engine)


class AreaObject:
    """
    A rectangular enclosure
    """

    def __init__(self, height, width):
        assert height > 2
        assert width > 2
        # note that the size is effectively height-2, width-2
        self.height = height
        self.width = width
        self.left_walls = []
        self.right_walls = []
        self.top_walls = []
        self.bottom_walls = []

    def place(self, position, world):
        # place walls
        xmin, ymin = position
        xmax = xmin + self.width
        ymax = ymin + self.height
        for x in range(xmin, xmax):
            w = O.Wall()
            w.place((x, ymin), world)
            self.top_walls.append(w)
            w = O.Wall()
            w.place((x, ymax-1), world)
            self.bottom_walls.append(w)
        for y in range(ymin+1, ymax-1):
            w = O.Wall()
            w.place((xmin, y), world)
            self.left_walls.append(w)
            w = O.Wall()
            w.place((xmax-1, y), world)
            self.right_walls.append(w)
        world.areas.add(self)

    def remove(self, world):
        world.areas.remove(self)
        for w in self.top_walls + self.bottom_walls + self.left_walls + self.right_walls:
            world.remove_object(w)


class Room(AreaObject):
    """
    An area with doors
    """

    def __init__(self, height, width, doors=None):
        super().__init__(height, width)
        self.doors = doors or []

    def place(self, position, world):
        super().place(position, world)
        xmin, ymin = position
        for door, door_pos in self.doors:
            x, y = door_pos
            new_pos = xmin+x, ymin+y
            walls = [o for o in world.get_objects_at_pos(
                new_pos) if isinstance(o, O.Wall)]
            for w in walls:
                world.remove_object(w)
            door.place(new_pos, world)

    def link(self, another, world):
        # find the closest doors
        best = None
        best_dist = np.inf
        for my_door, _ in self.doors:
            for another_door, _ in another.doors:
                dist = my_door.get_dist_to_position(another_door.position)
                if dist < best_dist:
                    best_dist = dist
                    best = my_door, another_door
        my_door, another_door = best
        my_door.link(another_door, world)
