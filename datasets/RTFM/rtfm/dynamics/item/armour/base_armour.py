# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
from ..base_item import BaseItem
from ... import material as M, element as E
from .. import modifier as O
from collections import defaultdict


class Armour(BaseItem):

    char = 'a'

    def __init__(
            self,
            name: str = None,
            weight: int = 0,
            armour_class: int = 0,
            material=M.Cloth,
    ):
        super().__init__(name=name, weight=weight, material=material)
        self.armour_class = armour_class
        self.elemental_armour_class = defaultdict(lambda: 0)

    @classmethod
    def get_random(cls, p_prefix=0.5, p_suffix=0.5, p_erosion=0.5):
        candidate_classes = [c for c in cls.get_all_subclasses(cls) if not c.__name__.startswith('Base')]
        C = random.choice(candidate_classes)
        inst = C()
        candidate_modifier_classes = [c for c in cls.get_all_subclasses(O.BaseArmourModifier) if not c.__name__.startswith('Base')]
        candidate_suffix_classes = [c for c in candidate_modifier_classes if issubclass(c, O.BaseSuffixModifier)]
        candidate_prefix_classes = [c for c in candidate_modifier_classes if issubclass(c, O.BasePrefixModifier)]

        if inst.material.erosion and random.random() < p_erosion:
            erosion_base = random.choice(list(inst.material.erosion))
            candidate_erosion_classes = cls.get_all_subclasses(erosion_base)
            erosion = random.choice(candidate_erosion_classes)
            inst.add_erosion(erosion)

        if random.random() < p_suffix:
            suffix = random.choice(candidate_suffix_classes)
            inst.add_suffix(suffix)

        for i in range(cls.max_prefix):
            if random.random() < p_prefix:
                prefix = random.choice(candidate_prefix_classes)
                inst.add_prefix(prefix)
        return inst

    def apply_modifier(self, modifier):
        modifier.apply_armour(self)
        return self

    def add_elemental_armour_class(self, element, ac):
        valid = E.Element.__subclasses__()
        assert element in valid, '{} is not a valid element: {}'.format(element, valid)
        self.elemental_armour_class[element] += ac
