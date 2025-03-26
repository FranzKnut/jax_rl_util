# Adapted from brax 2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Wrappers to support Brax and Gymnax training."""

from dataclasses import dataclass, field
from typing import Iterable

import brax
import gymnasium as gym
import gymnax
import popjym
from brax.envs.base import Env as BraxEnv  # noqa
from jax import numpy as jnp

from . import *  # noqa
from .dronegym import DroneGym
from .env_util import make_obs_mask
from .tribead import TriangleJax
from .wrappers import (
    EpisodeWrapper,
    FlatObsBraxWrapper,
    GymBraxWrapper,
    GymnaxBraxWrapper,
    POBraxWrapper,
    PopJymBraxWrapper,
    RandomizedAutoResetWrapper,
    VmapWrapper,
)


@dataclass(frozen=True, eq=True)
class EnvironmentConfig:
    """Parameters for gym environments.

    Attributes
    ----------
        obs_mask (Union[str, Iterable[int]]): Mask for the observation space.
        env_kwargs (dict): Arguments for the env step function.
        init_kwargs (dict): Initialization arguments for the environment.
        batch_size (int): Number of parallel environments.
        render (bool): Whether to render the environment during training.
    """

    env_name: str = "CartPole-v1"
    reward_scaling: int = 1
    obs_mask: str | Iterable[int] | None = None
    init_kwargs: dict = field(default_factory=dict, hash=False)
    env_kwargs: dict = field(default_factory=dict, hash=False)
    max_ep_length: int = 1000
    batch_size: int | None = None
    render: bool = True


def print_env_info(env_info):
    """Print infos about an environment. Takes env_info from make_env."""
    OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = env_info.values()
    print("ENV:")
    print(f"obs_size:    {OBS_SIZE}")
    print(f"act_size:    {ACT_SIZE}" + (" (discrete)" if DISCRETE else " (continuous)"))
    if isinstance(obs_mask, str):
        print(f"obs_mask:    {obs_mask}")
    elif len(obs_mask) < OBS_SIZE:
        print(f"obs_mask:    {obs_mask}")
    print(f"act_clip:    {act_clip}")
    # print(f'value_size: {VALUE_SIZE}')


def get_env_specs(env: gym.Env, obs_mask=None):
    """Infer the sizes for the observation and action space given a mask.

    Parameters
    ----------
    env : gym.Env
        _description_
    obs_mask : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    is_gymnax = hasattr(env, "observation_space")
    ACT_SIZE = env.action_size
    if is_gymnax:
        env: GymnaxBraxWrapper
        DISCRETE = env.discrete
        act_space = env.action_space
        if act_space.dtype == jnp.float32:
            if ACT_SIZE == 1:
                act_clip = (act_space.low, act_space.high)
            else:
                act_clip = tuple(map(tuple, (act_space.low._value, act_space.high._value)))
        else:
            act_clip = None
    else:
        env: BraxEnv
        # is brax
        DISCRETE = False
        # Assuming brax only takes normalized actions!
        act_clip = None
        act_clip = tuple([-1] * ACT_SIZE), tuple([1] * ACT_SIZE)
    obs_mask = make_obs_mask(env.observation_size, obs_mask)
    OBS_SIZE = len(obs_mask)

    return OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip


def make_env(
    params: EnvironmentConfig, debug=0, make_eval=False, use_vmap_wrapper=True
) -> tuple[BraxEnv, dict] | tuple[BraxEnv, dict, BraxEnv]:
    """Make brax or gymnax env.

    Parameters
    ----------
    params : EnvironmentParams
        additional params. must have 'env_name', 'batch_size' and 'max_ep_length'. 'obs_mask' is optional.
         Environment name. If starts with 'brax', will use brax env, otherwise gymnax.
    debug: int | bool
        set the debug level.
    make_eval : bool, optional
        Whether to make an eval env, by default False
        If true, eval env without batching is also returned
    use_vmap_wrapper : bool, optional
        Force using the vmap wrapper (even for batchsize 1), by default True

    Returns
    -------
    env : envs.Env
        Environment
    env_info : dict
        Dictionary with env info
    """
    env: BraxEnv
    env_name = params.env_name
    if env_name in gymnax.registered_envs:
        # Set params for gymnax envs
        params.env_kwargs["max_steps_in_episode"] = params.max_ep_length

        # create a gym environment
        env, gymnax_params = gymnax.make(env_name, **params.init_kwargs)
        env = GymnaxBraxWrapper(env, params.env_kwargs)
    elif env_name in popjym.registration.REGISTERED_ENVS:
        env, env_params = popjym.make(env_name)
        env = PopJymBraxWrapper(env, params.env_kwargs)
    elif "dronegym" in env_name.lower():
        env = DroneGym()
        env = GymnaxBraxWrapper(env, params.env_kwargs)
    elif "tribead" in env_name.lower():
        env = TriangleJax(**params.init_kwargs)
        env = GymnaxBraxWrapper(env, params.env_kwargs)
    elif env_name.startswith("brax-") or env_name in brax.envs._envs:
        # Create entrypoint for brax env
        env = brax.envs.get_environment(env_name=env_name.replace("brax-", ""), **params.env_kwargs)
    else:
        # Create a gym environment wrapped with vmap, AutoReset, and Episode wrappers
        env = gym.make(env_name, disable_env_checker=debug < 3, **params.init_kwargs)

        if not (env_name.startswith("brax-") or env_name in brax.envs._envs):
            # probably a gym env
            env = GymBraxWrapper(env, params.env_kwargs)

    OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = get_env_specs(env, params.obs_mask)

    # Wrap with the brax wrappers
    env = EpisodeWrapper(env, params.max_ep_length, action_repeat=1)
    env = FlatObsBraxWrapper(env)
    if obs_mask is not None:
        env = POBraxWrapper(env, obs_mask)
    env = RandomizedAutoResetWrapper(env)
    # env = EfficientAutoResetWrapper(env)
    if (params.batch_size is not None and (params.batch_size > 1)) or use_vmap_wrapper:
        env = VmapWrapper(env, batch_size=params.batch_size)
    env_info = dict(obs_size=OBS_SIZE, discrete=DISCRETE, act_size=ACT_SIZE, obs_mask=obs_mask, act_clip=act_clip)

    if make_eval:
        if env_name in gymnax.registered_envs:
            eval_env, _ = gymnax.make(env_name, **params.init_kwargs)
            eval_env = GymnaxBraxWrapper(eval_env, params.env_kwargs)
        elif "dronegym" in env_name.lower():
            eval_env = DroneGym(**params.init_kwargs)
            eval_env = GymnaxBraxWrapper(eval_env, params.env_kwargs)
        elif "tribead" in env_name.lower():
            eval_env = TriangleJax(**params.init_kwargs)
            eval_env = GymnaxBraxWrapper(eval_env, params.env_kwargs)
        elif env_name in popjym.registration.REGISTERED_ENVS:
            eval_env, _ = popjym.make(env_name)
            eval_env = PopJymBraxWrapper(eval_env, params.env_kwargs)
        elif env_name.startswith("brax-") or env_name in brax.envs._envs:
            # Create entrypoint for brax env
            eval_env = brax.envs.get_environment(env_name=env_name.replace("brax-", ""), **params.env_kwargs)
        else:
            eval_env = gym.make(env_name, disable_env_checker=getattr(params, "debug", 0) < 3, **params.init_kwargs)
            eval_env = GymBraxWrapper(eval_env, params.env_kwargs)

        eval_env = EpisodeWrapper(eval_env, params.max_ep_length, action_repeat=1)
        eval_env = FlatObsBraxWrapper(eval_env)
        if obs_mask is not None:
            eval_env = POBraxWrapper(eval_env, obs_mask)
        eval_env = RandomizedAutoResetWrapper(eval_env)
        return env, env_info, eval_env

    return env, env_info
