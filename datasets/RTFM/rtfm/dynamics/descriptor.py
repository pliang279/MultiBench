# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random


class Descriptor:

    def __init__(self, adj, range):
        self.adj = adj
        self.range = range

    def contains_value(self, val):
        return val in self.range

    def sample(self):
        return random.choice(self.range)


class ConstDescriptor(Descriptor):

    def __init__(self, adj):
        super().__init__(adj, range=None)

    def contains_value(self, val):
        return self.adj == val

    def sample(self):
        return self.adj


class NumDescriptor(Descriptor):

    def contains_value(self, val):
        low, high = self.range
        return low <= val <= high

    def sample(self):
        low, high = self.range
        return random.uniform(low, high)


class IntDescriptor(NumDescriptor):

    def contains_value(self, val):
        low, high = self.range
        return low <= val < high

    def sample(self):
        low, high = self.range
        return random.randint(low, high-1)


class DescriptorCollection(list):

    def val_to_description(self, val):
        d = self[0]
        for d in self:
            if d.contains_value(val):
                break
        return d.adj

    def sample(self):
        return random.choice(self).sample()
