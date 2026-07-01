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

import brax
import gymnasium as gym
from jax import numpy as jnp
import numpy as np

from jax_rl_util.envs import wrappers
from jax_rl_util.envs.wrappers import Env

# Try importing optional dependencies
try:
    import mujoco_playground

    MUJOCO_PLAYGROUND_INSTALLED = True
except ImportError:
    MUJOCO_PLAYGROUND_INSTALLED = False
    print("mujoco_playground not installed. Skipping mujoco_playground envs.")

try:
    import popjym

    POPJYM_INSTALLED = True
except ImportError:
    POPJYM_INSTALLED = False
    print("popjym not installed. Skipping popjym envs.")

try:
    import gymnax

    GYMNAX_INSTALLED = True
except:
    GYMNAX_INSTALLED = False
    print("gymnax not installed. Skipping gymnax envs.")

try:
    import highway_env  # noqa

    HIGHWAY_ENV_INSTALLED = True
except ImportError:
    HIGHWAY_ENV_INSTALLED = False
    print("highway_env not installed. Skipping highway_env envs.")

try:
    from .carracing_penalty import CarRacingPenaltyEnv  # noqa

    BOX2D_INSTALLED = True
except ImportError:
    BOX2D_INSTALLED = False
    print("Box2D not installed. Skipping CarRacingPenaltyEnv.")


from . import *  # noqa
from .dronegym import DroneGym
from .continuous_cartpole import ContinuousCartPoleEnv
from .env_util import make_obs_mask
from .tribead import TriangleJax
from .wrappers import (
    EpisodeWrapper,
    FlatObsBraxWrapper,
    GrayscaleWrapper,
    GymBraxWrapper,
    GymJaxWrapper,
    GymnaxBraxWrapper,
    POBraxWrapper,
    PopJymBraxWrapper,
    RandomizedAutoResetWrapper,
    VmapWrapper,
)

BRAX_ENVS_POS_DIMS = {"ant": 2, "halfcheetah": 1, "humanoid": 2}


@dataclass(frozen=True, eq=True)
class EnvironmentConfig:
    """Parameters for gym environments.

    Attributes
    ----------
        env_name (str): Environment name. Supported are brax, gymnax, popjym, highway_env, mujoco_playgound and gym envs.
        obs_mask (Union[str, tuple[int]]): Mask for the observation space.
        init_kwargs (dict): Initialization arguments for the environment.
        env_kwargs (dict): Arguments for the env step function.
        max_ep_length (int): Maximum episode length.
        batch_size (int): Number of parallel environments.
        transform_wrappers (list): List of wrappers to apply to the environment.
    """

    env_name: str = "CartPole-v1"
    # reward_scaling: int = 1
    obs_mask: str | tuple[int] | None = None
    init_kwargs: dict = field(default_factory=dict, hash=False)
    step_kwargs: dict = field(default_factory=dict, hash=False)
    max_ep_length: int = 1000
    batch_size: int | None = None
    transform_wrappers: list = field(default_factory=list, hash=False)

    @property
    def env_kwargs(self):
        """For backwards compatibility."""
        return self.step_kwargs


def print_env_info(env_info):
    """Print infos about an environment. Takes env_info from make_env."""
    env_name, package, OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = (
        env_info.values()
    )
    print(f"ENV:         {env_name}")
    print(f"package:     {package}")
    print(f"obs_size:    {OBS_SIZE}")
    print(f"obs_size:    {OBS_SIZE}")
    print(f"act_size:    {ACT_SIZE}" + (" (discrete)" if DISCRETE else " (continuous)"))
    print(f"obs_mask:    {obs_mask}")
    print(f"act_clip:    {act_clip}")
    # print(f'value_size: {VALUE_SIZE}')


def get_env_specs(env: Env, obs_mask=None):
    """Infer the sizes for the observation and action space given a mask."""
    is_gym = hasattr(env, "observation_space")
    ACT_SIZE = env.action_size
    if is_gym:
        env: GymnaxBraxWrapper
        DISCRETE = env.discrete
        act_space = env.action_space
        if act_space.dtype == jnp.float32:
            if ACT_SIZE == 1:
                act_clip = (act_space.low, act_space.high)
            else:
                act_clip = tuple(
                    map(tuple, (np.array(act_space.low), np.array(act_space.high)))
                )
        else:
            act_clip = None
    else:
        env: Env
        # is brax
        DISCRETE = False
        # Assuming brax only takes normalized actions!
        act_clip = None
        act_clip = tuple([-1] * ACT_SIZE), tuple([1] * ACT_SIZE)
    obs_mask = make_obs_mask(env.observation_size, obs_mask)
    OBS_SIZE = len(obs_mask)

    return OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip


def get_env(config: EnvironmentConfig, debug=0) -> gym.Env:
    """Get an env from config.

    Parameters
    ----------
    config : EnvironmentConfig
        Environment configuration.

    Returns
    -------
    envs.Env
        Environment.
    """
    env_name = config.env_name
    if GYMNAX_INSTALLED and env_name in gymnax.registered_envs:
        # Set params for gymnax envs
        config.step_kwargs["max_steps_in_episode"] = config.max_ep_length

        # create a gym environment
        env, gymnax_params = gymnax.make(env_name, **config.init_kwargs)
        env = GymnaxBraxWrapper(env, config.step_kwargs)
        env.package_name = "gymnax"
    elif POPJYM_INSTALLED and env_name in popjym.registration.REGISTERED_ENVS:
        env, env_params = popjym.make(env_name)
        env = PopJymBraxWrapper(env, config.step_kwargs)
        env.package_name = "popjym"
    elif "dronegym" in env_name.lower():
        env = DroneGym(**config.init_kwargs)
        env = GymnaxBraxWrapper(env, config.step_kwargs)
        env.package_name = "misc"
    elif "tribead" in env_name.lower():
        env = TriangleJax(**config.init_kwargs)
        env = GymnaxBraxWrapper(env, config.step_kwargs)
        env.package_name = "misc"
    elif env_name.startswith("brax-") or env_name in brax.envs._envs:
        # Create entrypoint for brax env
        env = brax.envs.get_environment(
            env_name=env_name.replace("brax-", ""), **config.init_kwargs
        )
        env.package_name = "brax"

    elif MUJOCO_PLAYGROUND_INSTALLED and (
        env_name.startswith("playground-")
        or env_name in mujoco_playground.registry.ALL_ENVS
    ):
        env = mujoco_playground.registry.load(env_name.replace("playground-", ""))
        # env = GymnaxBraxWrapper(env, params.env_kwargs)
        env.package_name = "mujoco_playground"
    else:
        # Create gym environment
        env = gym.make(
            env_name.replace("gym-", ""),
            disable_env_checker=debug < 3,
            **config.init_kwargs,
        )
        if "cartpolecontinuousjax" in env_name.lower():
            # CartpoleContinuousJaxSwingUp is a custom env that does not need a JaxWrapper
            env = GymnaxBraxWrapper(env, config.step_kwargs)
            env.package_name = "misc"
        else:
            env = GymJaxWrapper(env)
            env = GymBraxWrapper(env, config.step_kwargs)
            env.package_name = "gym"

    # Make sure it knows its name for compatibility with other packages
    env.env_name = env_name
    env.name = env_name
    return env


def make_wrapped_env(
    config: EnvironmentConfig,
    debug=0,
    make_eval=False,
    autoreset=True,
    use_vmap_wrapper=True,
    extra_wrappers: list | None = None,
) -> tuple[Env, dict] | tuple[Env, dict, Env]:
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
    autoreset : bool, optional
        Whether to automatically reset the environment, by default True
    use_vmap_wrapper : bool, optional
        Force using the vmap wrapper (even for batchsize 1), by default True
    extra_wrappers : list, optional
        List of (wrapper_class, kwargs) tuples to apply after VmapWrapper.
        Example: [(SuddenNoiseWrapper, {'noise_strength': 1.0, 'sudden_noise_start': 5})]

    Returns
    -------
    env : envs.Env
        Environment
    env_info : dict
        Dictionary with env info
    eval_env : envs.Env, optional
        Eval environment (only if make_eval is True)
    """
    # TODO refactor:
    # [ ] Make env_info a field of the env.

    env: Env
    env_name = config.env_name

    env = get_env(config)
    OBS_SIZE, DISCRETE, ACT_SIZE, obs_mask, act_clip = get_env_specs(
        env, config.obs_mask
    )
    env.name = env_name

    for w in config.transform_wrappers:
        env = getattr(wrappers, w)(env)

    # Wrap with the brax wrappers
    env = EpisodeWrapper(env, config.max_ep_length, action_repeat=1)
    # env = FlatObsBraxWrapper(env)
    if config.obs_mask is not None:
        env = POBraxWrapper(env, config.obs_mask)

    # Autoreset
    if autoreset:
        env = RandomizedAutoResetWrapper(env)
    # env = EfficientAutoResetWrapper(env)

    # Apply extra wrappers (e.g., SuddenNoiseWrapper)
    if extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    # Use VmapWrapper for batching if batch_size > 1 or if use_vmap_wrapper is True
    if (config.batch_size is not None and (config.batch_size > 1)) or use_vmap_wrapper:
        env = VmapWrapper(env, batch_size=config.batch_size)

    env_info = dict(
        env_name=env_name,
        package_name=env.package_name,
        obs_size=OBS_SIZE,
        discrete=DISCRETE,
        act_size=ACT_SIZE,
        obs_mask=config.obs_mask,
        act_clip=act_clip,
    )

    if make_eval:
        # Eval env is the same as above but without batching
        eval_env = get_env(config=config, debug=debug)
        eval_env.name = env_name
        for w in config.transform_wrappers:
            eval_env = getattr(wrappers, w)(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.max_ep_length, action_repeat=1)
        # eval_env = FlatObsBraxWrapper(eval_env)
        if config.obs_mask is not None:
            eval_env = POBraxWrapper(eval_env, config.obs_mask)
        if autoreset:
            eval_env = RandomizedAutoResetWrapper(eval_env)
        # Apply extra wrappers to eval env too
        if extra_wrappers is not None:
            for wrapper_cls, wrapper_kwargs in extra_wrappers:
                eval_env = wrapper_cls(eval_env, **wrapper_kwargs)
        return env, env_info, eval_env

    return env, env_info
