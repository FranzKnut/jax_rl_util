"""Utiliy functions for working with environments."""

from typing import Callable, Iterable, Literal
from functools import partial

import brax.envs
import gymnasium as gym
import jax
import numpy as np
from jax import numpy as jnp

from jax_rl_util.util.logging_util import tree_stack


@partial(jax.jit, static_argnames=["agg_fn", "mode"])
def compute_agg_reward(
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    agg_fn: Callable | None = jnp.mean,
    carry_over_reward: jnp.ndarray | None = None,
    mode: Literal["first", "mean"] = "first",
):
    """Compute the average reward per episode from a batch of trajectories.

    Parameters
    ----------
    rewards : jnp.ndarray
        A batch of rewards from a Brax environment with shape (T, B), where T is the number of timesteps and B is the batch size.
    dones : jnp.ndarray
        A batch of done flags from a Brax environment with shape (T, B), where T is the number of timesteps and B is the batch size.
    agg_fn : callable, optional
        A function to aggregate the rewards, by default jnp.mean.
    carry_over_reward : jnp.ndarray, optional
        Sometimes we want to compute the reward for an episode that has just ended but the first part is not in states.
        The reward carried over from earlier can be given for each episode in the batch as tensor of shape (B,).
        If carry_over_reward is not None, the function retursn also the next carry_over_reward in second position.
    mode : str, optional
        Mode of computing the reward. Options are 'first' (default) or 'mean'.
        'first' computes the reward for the first episode in each batch dimension,
        while 'mean' computes the mean reward across all episodes in each batch dimension.
    """
    # For episodes that are done early, get the first occurence of done
    _dones = dones.cumsum(axis=0) if mode == "mean" else dones
    ep_until = jnp.where(
        dones.any(axis=0),
        _dones.argmax(axis=0),
        dones.shape[0],
    )
    # Compute cumsum and get value corresponding to end of episode per batch.
    # mean_reward = jnp.sum(traj_batch.reward) / jnp.max(jnp.array([jnp.sum(traj_batch.done), 1]))
    reward_per_dim = rewards.cumsum(axis=0)[ep_until, jnp.arange(ep_until.shape[-1])]
    if carry_over_reward is not None:
        carry_over_reward = rewards.cumsum(axis=0)[-1] - reward_per_dim
        reward_per_dim = reward_per_dim + carry_over_reward

    if mode == "mean":
        reward_per_dim = reward_per_dim / jnp.clip(dones.sum(axis=0), min=1)

    if agg_fn is not None:
        reward_per_dim = agg_fn(reward_per_dim)
    if carry_over_reward is not None:
        return reward_per_dim, carry_over_reward
    return reward_per_dim


def render_brax(env, states, render_steps=100, render_start=0, camera=None):
    """Render a sequence of states from a Brax environment.
    Parameters
    ----------
    env : brax.envs.Env
        The Brax environment to render.
    states : brax.envs.State
        The states to render, typically from a batch of trajectories.
    render_steps : int, optional
        Number of steps to render, by default 100.
    render_start : int, optional
        Start rendering from this step, by default 0.
    camera : str or int, optional
        Camera to use for rendering. If None, uses 'track' camera if available, otherwise
        uses the first camera. If an integer, it specifies the camera index.
    Returns
    -------
    np.ndarray
        Rendered image as a numpy array of shape (height, width, 3).
    """
    from brax.io import image

    steps = len(states.pipeline_state.q)
    states_to_render = [
        jax.tree.map(lambda x: x[n], states.pipeline_state)
        for n in range(steps)
        if n > render_start and n < render_start + render_steps
    ]
    camera = camera or ("track" if len(env.sys.cam_bodyid) else -1)
    return image.render_array(env.sys, states_to_render, camera=camera)


def make_obs_mask(
    base_obs_size: int, obs_mask: Iterable[int] | str | int | None = None
):
    """Get the observation mask from string description.

    obs_mask may take values ['odd', 'even', 'first_half', 'second_half'] or a list of indices.
    """
    # Flat observation size
    if not isinstance(base_obs_size, int):
        base_obs_size = np.prod(base_obs_size)
    if obs_mask == "odd" or obs_mask == "even":
        obs_mask = [i for i in range(base_obs_size) if i % 2 == (obs_mask == "odd")]
    elif obs_mask == "first_half":
        obs_mask = [i for i in range((base_obs_size) // 2)]  # Rounding down
    elif obs_mask == "second_half":
        obs_mask = [i for i in range((base_obs_size) // 2, base_obs_size)]
    elif isinstance(obs_mask, int):
        obs_mask = jnp.arange(base_obs_size, dtype=jnp.int32)
    elif obs_mask is None or (isinstance(obs_mask, str) and obs_mask.lower() == "none"):
        obs_mask = jnp.arange(base_obs_size, dtype=jnp.int32)
    return jnp.array(obs_mask, dtype=jnp.int32)


def render_frames(
    env: gym.Env,
    states: list,
    start_idx: int = None,
    end_idx: int = None,
):
    """Render the given states of the environment.

    Parameters
    ----------
    env : gym.Env
        Environment to render. Can handle Brax, Gymnax and Gym envs.
    states: list
        List of states to render.
    start_idx : int, optional
        start rendering from this index, by default None, means start at 0
    end_idx : int, optional
        render until this index, by default None, means render all

    Returns
    -------
    list[array]
        List of RGB array renderings of the environment at given states.
    """
    if not isinstance(states, list):
        states = [
            jax.tree.map(lambda x: x[n], states)
            for n in range(start_idx or 0, end_idx or states.done.shape[0])
        ]
    from jax_rl_util.envs.wrappers import GymnaxBraxWrapper

    frames = []
    try:
        is_brax = env.name.startswith("brax-") or env.name in brax.envs._envs
        if env.name == "dronegym":
            from jax_rl_util.envs.plot_drones import plot_drones

            states = tree_stack(states)
            data = states.pipeline_state
            data["reward"] = states.reward
            data["done"] = states.done[
                1:
            ]  # shift by 1 since 'done' always marks the obs after reset
            return plot_drones(env.params, data, obstacle=env.obstacle)
        else:
            try:
                # Try to import mujoco_playground
                import mujoco_playground

                # Define rendering function for specific envs
                is_playground = (
                    env.name.startswith("playground-")
                    or env.name in mujoco_playground.registry.ALL_ENVS
                )
            except ImportError:
                is_playground = False
            if not is_playground:
                states = [x.pipeline_state for x in states]
            if is_brax and len(states[0].q.shape) >= 2:
                states = jax.tree.map(lambda x: x[0], states)

        if is_playground:
            return env.render(states)
        elif env is not None and isinstance(env.unwrapped, GymnaxBraxWrapper):
            from gymnax.visualize.vis_gym import get_gym_state

            _env_name = env.name
            if "CartPole" in _env_name:
                _env_name = "CartPole-v1"

            gym_env = gym.make(_env_name, render_mode="rgb_array").unwrapped

            def render_gym(_state):
                """Taken from gymnax.visualize.vis_gym."""
                gym_state = get_gym_state(_state, _env_name)
                if _env_name == "Pendulum-v1":
                    gym_env.last_u = gym_state[-1]
                gym_env.state = gym_state
                return gym_env.render()

        elif is_brax:
            from brax.io import image

            def render_gym(_state):
                camera = "track" if len(env.sys.cam_bodyid) else -1
                camera = "track" if "inverted_pendulum" not in env.name else None
                return image.render_array(
                    env.sys, _state, 256, 256, camera=camera
                )  # .transpose(2, 0, 1)
        else:
            gym_env = gym.make(env.name, render_mode="rgb_array").unwrapped

            def render_gym(_state):
                if env.name in ["CarRacing-v3", "CarRacingPenalty-v0"]:
                    return _state
                gym_env.state = _state
                if env.name == "Pendulum-v1":
                    gym_env.unwrapped.env.last_u = _state[-1]
                return gym_env.render()  # .transpose(2, 0, 1)

        for _state in states:
            if is_brax and len(_state.q.shape) >= 2:
                _state = jax.tree.map(lambda x: x[0], _state)
            frames.append(render_gym(_state))

        if isinstance(env.unwrapped, GymnaxBraxWrapper) or not is_brax:
            gym_env.close()
    except Exception as e:
        print(f"Rendering failed with error: {e}")
    return frames
