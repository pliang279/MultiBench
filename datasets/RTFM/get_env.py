"""Implements environment getter for RTFM."""
import gym
from rtfm import featurizer as X
import rtfm.tasks


def create_env(env="rtfm:groups_simple_stationary-v0", height=6, width=6, partially_observable=False, max_placement=1, featurizer=None, shuffle_wiki=False, time_penalty=-0.02):
    """Create RTFM environment.

    Args:
        env (str, optional): RTFM environment name.. Defaults to "rtfm:groups_simple_stationary-v0".
        height (int, optional): Height of environment. Defaults to 6.
        width (int, optional): Width of environment. Defaults to 6.
        partially_observable (bool, optional): Whether to only give partial observations or priviledged observations. Defaults to False.
        max_placement (int, optional): Max placement. Defaults to 1.
        featurizer (_type_, optional): Function for featurizing inputs. Defaults to None.
        shuffle_wiki (bool, optional): Whether to shuffle wiki. Defaults to False.
        time_penalty (float, optional): Time penalty. Defaults to -0.02.

    Returns:
        env: gym environment
    """
    f = featurizer or X.Concat([X.Text(), X.ValidMoves()])
    env = gym.make(env, room_shape=(height, width), partially_observable=partially_observable,
                   max_placement=max_placement, featurizer=f, shuffle_wiki=shuffle_wiki, time_penalty=time_penalty)
    return env
