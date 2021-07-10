# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random


class Dice:

    def __init__(self):
        pass

    def roll(self):
        raise NotImplementedError()

    def describe(self):
        raise NotImplementedError()

    @classmethod
    def from_str(cls, s):
        dice = []
        for sub in s.split('+'):
            sub = sub.strip()
            if 'd' not in sub:
                constant = int(sub)
                dice.append(ConstantDice(constant))
            else:
                splits = sub.split('d')
                if len(splits) == 1:
                    num = 1
                    max_roll = splits[0]
                elif len(splits) == 2:
                    num, max_roll = splits
                    if num == '':
                        num = 1
                else:
                    raise Exception('could not parse dice string {} in main dice string {}'.format(sub, s))
                dice.extend([SingleDice(max=int(max_roll)) for _ in range(int(num))])
        return SumDice(dice) if len(dice) > 1 else dice[0]


class ConstantDice(Dice):

    def __init__(self, constant):
        super().__init__()
        self.constant = self.max = constant

    def roll(self):
        return self.constant

    def describe(self):
        return repr(self.constant)


class SingleDice(Dice):

    def __init__(self, max=20):
        super().__init__()
        self.max = max

    def roll(self):
        return random.randint(1, self.max)

    def describe(self):
        return 'd{}'.format(self.max)


class SumDice(Dice):

    def __init__(self, subdice):
        super().__init__()
        self.sub = subdice

    @property
    def max(self):
        return sum(d.max for d in self.sub)

    def roll(self):
        return sum(d.roll() for d in self.sub)

    def describe(self):
        return ' + '.join([d.describe() for d in self.sub])
