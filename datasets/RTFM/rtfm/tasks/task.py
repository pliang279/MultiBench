# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
from vocab import Vocab
from rtfm import featurizer as F, utils
from rtfm.dynamics import monster as M, item as I, world as W, engine as E, world_object as O


class Task(gym.Env):

    default_agent_criteria = {k: v[-1] for k, v in M.Agent.descriptors.items()}

    def __init__(self, featurizer=F.Progress(), partially_observable=False, max_iter=1000, max_placement=2, max_name=2, max_inv=10, max_wiki=80, max_task=40, time_penalty=-.02, shuffle_wiki=False):
        self.world = W.World()
        self.engine = E.Engine()
        self.partially_observable = partially_observable
        self.history = []
        self.iter = 0
        self.max_iter = max_iter
        self.max_placement = max_placement
        self.max_name = max_name
        self.max_inv = max_inv
        self.max_wiki = max_wiki
        self.max_task = max_task
        self.time_penalty = time_penalty
        self.shuffle_wiki = shuffle_wiki

        self.renderer = F.Terminal()
        self.featurizer = featurizer
        self.agent = M.QueuedAgent()

        # action space
        self.action_space = M.QueuedAgent.valid_moves

        # observation shapes
        self.observation_space = self.featurizer.get_observation_space(self)

        self.vocab = Vocab(['pad', 'eos', ''])
        self.build_vocab()
        self.reset()

    @property
    def perspective(self):
        return self.agent if self.partially_observable else None

    def add_words(self, words):
        self.vocab.word2index(list(words), train=True)

    def build_vocab(self):
        self.add_words([c.__name__.split('.')[-1].lower()
                       for c in utils.get_all_subclasses(O.WorldObject)])

    def agent_is_dead(self):
        return not self.agent.is_alive()

    def out_of_turns(self):
        return self.iter >= self.max_iter

    def get_reward_finish_win(self):
        raise NotImplementedError()

    def step(self, Action):
        if isinstance(Action, int):
            Action = M.QueuedAgent.valid_moves[Action]
        self.iter += 1
        if self.agent is not None:
            self.agent.queue_action(Action)
        executed = self.engine.run_turn(self.world)
        self.history.append(executed)
        r, f, w = self.get_reward_finish_win()
        return self.featurizer.featurize(self), r, f, w

    def get_world(self):
        raise NotImplementedError()

    def get_inv(self):
        return self.agent.inventory.describe()

    def get_wiki(self):
        raise NotImplementedError()

    def get_task(self):
        raise NotImplementedError()

    def reset(self):
        self._reset()
        return self.featurizer.featurize(self)

    def _reset(self):
        self.iter = 0
        self.world.reset()
        self.engine.reset()
        self.history.clear()

    def close(self):
        pass
