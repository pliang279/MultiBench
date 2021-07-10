# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .item.modifier import erosion as E


class Material:
    erosion = {}

    @classmethod
    def describe(cls):
        return cls.__name__.lower()


class Copper(Material):
    erosion = {E.BaseCorrosion}


class Iron(Material):
    erosion = {E.BaseRust, E.BaseCorrosion}


class Silver(Material):
    pass


class Wood(Material):
    erosion = {E.BaseBurn, E.BaseRot}


class Mineral(Material):
    pass


class Metal(Material):
    pass


class Leather(Material):
    erosion = {E.BaseBurn, E.BaseRot}


class Plastic(Material):
    erosion = {E.BaseBurn}


class Bone(Material):
    pass


class Flesh(Material):
    pass


class Cloth(Material):
    erosion = {E.BaseBurn, E.BaseRot}


class Glass(Material):
    pass


class Dragon(Material):
    pass


class Mithril(Material):
    pass


class Undefined(Material):
    pass
