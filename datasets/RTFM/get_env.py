import gym
from rtfm import featurizer as X
import rtfm.tasks

def create_env(env="rtfm:groups_simple_stationary-v0", height=6, width=6, partially_observable=False, max_placement=1, featurizer=None, shuffle_wiki=False, time_penalty=-0.02):
    f = featurizer or X.Concat([X.Text(), X.ValidMoves()])
    env = gym.make(env, room_shape=(height, width), partially_observable=partially_observable, max_placement=max_placement, featurizer=f, shuffle_wiki=shuffle_wiki, time_penalty=time_penalty)
    return env