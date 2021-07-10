# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class Element:

    @classmethod
    def describe(cls):
        return cls.__name__.lower()


class Cold(Element):
    pass


class Fire(Element):
    pass


class Lightning(Element):
    pass


class Poison(Element):
    pass
