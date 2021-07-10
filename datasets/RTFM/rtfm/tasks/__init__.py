# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .rock_paper_scissors import RockPaperScissors, RockPaperScissorsDev, RockPaperScissorsMed, RockPaperScissorsMedDev, RockPaperScissorsHard, RockPaperScissorsHardDev
from .groups import Groups, GroupsDev, GroupsStationary, GroupsStationaryDev, GroupsSimple, GroupsSimpleDev, GroupsSimpleStationary, GroupsSimpleStationaryDev, GroupsSimpleStationarySingleMonster, GroupsSimpleStationarySingleMonsterDev, GroupsSimpleStationarySingleItem, GroupsSimpleStationarySingleItemDev, GroupsNL, GroupsNLDev, GroupsSimpleNL, GroupsSimpleNLDev, GroupsStationaryNL, GroupsStationaryNLDev, GroupsSimpleStationaryNL
from gym.envs.registration import register as register_env


register_env(
    id='rock_paper_scissors-v0',
    entry_point='rtfm.tasks:RockPaperScissors',
)

register_env(
    id='rock_paper_scissors_dev-v0',
    entry_point='rtfm.tasks:RockPaperScissorsDev',
)

register_env(
    id='rock_paper_scissors_med-v0',
    entry_point='rtfm.tasks:RockPaperScissorsMed',
)

register_env(
    id='rock_paper_scissors_med_dev-v0',
    entry_point='rtfm.tasks:RockPaperScissorsMedDev',
)

register_env(
    id='rock_paper_scissors_hard-v0',
    entry_point='rtfm.tasks:RockPaperScissorsHard',
)

register_env(
    id='rock_paper_scissors_hard_dev-v0',
    entry_point='rtfm.tasks:RockPaperScissorsHardDev',
)

register_env(
    id='groups-v0',
    entry_point='rtfm.tasks:Groups',
)

register_env(
    id='groups_dev-v0',
    entry_point='rtfm.tasks:GroupsDev',
)

register_env(
    id='groups_stationary-v0',
    entry_point='rtfm.tasks:GroupsStationary',
)

register_env(
    id='groups_stationary_dev-v0',
    entry_point='rtfm.tasks:GroupsStationaryDev',
)

register_env(
    id='groups_simple-v0',
    entry_point='rtfm.tasks:GroupsSimple',
)

register_env(
    id='groups_simple_dev-v0',
    entry_point='rtfm.tasks:GroupsSimpleDev',
)

register_env(
    id='groups_simple_stationary-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationary',
)

register_env(
    id='groups_simple_stationary_dev-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationaryDev',
)

register_env(
    id='groups_simple_stationary_single_monster-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationarySingleMonster',
)

register_env(
    id='groups_simple_stationary_single_monster_dev-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationarySingleMonsterDev',
)

register_env(
    id='groups_simple_stationary_single_item-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationarySingleItem',
)

register_env(
    id='groups_simple_stationary_single_item_dev-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationarySingleItemDev',
)

register_env(
    id='groups_nl-v0',
    entry_point='rtfm.tasks:GroupsNL',
)

register_env(
    id='groups_nl_dev-v0',
    entry_point='rtfm.tasks:GroupsNLDev',
)

register_env(
    id='groups_simple_nl-v0',
    entry_point='rtfm.tasks:GroupsSimpleNL',
)

register_env(
    id='groups_simple_nl_dev-v0',
    entry_point='rtfm.tasks:GroupsSimpleNLDev',
)

register_env(
    id='groups_stationary_nl-v0',
    entry_point='rtfm.tasks:GroupsStationaryNL',
)

register_env(
    id='groups_stationary_nl_dev-v0',
    entry_point='rtfm.tasks:GroupsStationaryNLDev',
)

register_env(
    id='groups_simple_stationary_nl-v0',
    entry_point='rtfm.tasks:GroupsSimpleStationaryNL',
)
