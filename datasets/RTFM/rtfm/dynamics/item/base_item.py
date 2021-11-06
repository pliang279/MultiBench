# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from rtfm.dynamics import world_object as O


class BaseItem(O.WorldObject):
    max_prefix = 3

    def __init__(self, name=None, weight=0, material=None):
        super().__init__(name)
        self.weight = weight
        self.prefixes = []
        self.suffix = None
        self.material = material
        self.erosion = {}

    def add_erosion(self, value):
        assert self.material is not None
        for e in self.material.erosion:
            if issubclass(value, e):
                self.erosion[e] = value
                self.apply_modifier(value)
                return
        raise Exception(
            'Item {} does not support erosion {}'.format(self, value))

    def add_prefix(self, modifier):
        assert len(self.prefixes) < self.max_prefix, '{} already has max prefix of {}: {}'.format(
            self, self.max_prefix, self.prefixes)
        self.prefixes.append(modifier)
        self.apply_modifier(modifier)
        return self

    def add_suffix(self, modifier):
        assert self.suffix is None, '{} already has suffix of {}'.format(
            self, self.suffix)
        self.suffix = modifier
        self.apply_modifier(modifier)
        return self

    def apply_modifier(self, modifier):
        raise NotImplementedError()

    def describe(self):
        prefix = ' '.join([m.describe() for m in self.prefixes])
        suffix = '' if self.suffix is None else 'of {}'.format(
            self.suffix.describe())
        s = '{} {} {}'.format(prefix, self.name, suffix).strip()
        if self.erosion:
            s = '{} {}'.format(' '.join([v.describe()
                               for v in self.erosion.values()]), s)
        return s
