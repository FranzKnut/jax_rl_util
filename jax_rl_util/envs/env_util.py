"""Utiliy functions for working with environments."""

from typing import Iterable

import brax.envs
import gymnasium as gym
import jax
import numpy as np
from jax import numpy as jnp
from typing_extensions import deprecated

from jax_rl_util.envs.plot_drones import plot_drones
from jax_rl_util.util.logging_util import tree_stack


def compute_agg_reward(states: brax.envs.State, agg_fn=jnp.mean):
    """Compute the average reward per episode from a batch of trajectories."""
    # For episodes that are done early, get the first occurence of done
    ep_until = jnp.where(
        states.done.any(axis=0), states.done.argmax(axis=0), states.done.shape[0]
    )
    # Compute cumsum and get value corresponding to end of episode per batch.
    # mean_reward = jnp.sum(traj_batch.reward) / jnp.max(jnp.array([jnp.sum(traj_batch.done), 1]))
    return agg_fn(
        states.reward.cumsum(axis=0)[ep_until, jnp.arange(ep_until.shape[-1])]
    )


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


@deprecated("Deprecated for Brax Envs. Will be removed in the future.")
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
    _env: gym.Env, states: list, start_idx: int = None, end_idx: int = None
):
    """Render the given states of the environment.

    Parameters
    ----------
    _env : gym.Env
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
            for n in range(start_idx or 0, end_idx or states.time.shape[0])
        ]
    from jax_rl_util.envs.wrappers import GymnaxBraxWrapper

    frames = []
    try:
        import mujoco_playground
        # Define rendering function for specific envs
        is_playground = (
            _env.name.startswith("playground-")
            or _env.name in mujoco_playground.registry.ALL_ENVS
        )
        is_brax = _env.name.startswith("brax-") or _env.name in brax.envs._envs
        if _env.name == "dronegym":
            states = tree_stack(states)
            data = states.pipeline_state
            data["reward"] = states.reward
            data["done"] = states.done[
                1:
            ]  # shift by 1 since 'done' always marks the obs after reset
            return plot_drones(_env.params, data, obstacle=_env.obstacle)
        else:
            if not is_playground:
                states = [x.pipeline_state for x in states]
            states = jax.tree.map(lambda x: x[0], states)

        if is_playground:
            return _env.render(states)
        elif isinstance(_env.unwrapped, GymnaxBraxWrapper):
            from gymnax.visualize.vis_gym import get_gym_state

            gym_env = gym.make(_env.name, render_mode="rgb_array").unwrapped

            def render_gym(_state):
                """Taken from gymnax.visualize.vis_gym."""
                gym_state = get_gym_state(_state, _env.name)
                if _env.name == "Pendulum-v1":
                    gym_env.last_u = gym_state[-1]
                gym_env.state = gym_state
                return gym_env.render()

        elif is_brax:
            from brax.io import image

            def render_gym(_state):
                camera = "track" if len(_env.sys.cam_bodyid) else -1
                camera = "track" if "inverted_pendulum" not in _env.name else None
                return image.render_array(
                    _env.sys, _state, 256, 256, camera=camera
                )  # .transpose(2, 0, 1)
        else:
            gym_env = gym.make(_env.name, render_mode="rgb_array").unwrapped

            def render_gym(_state):
                gym_env.state = _state
                if _env.name == "Pendulum-v1":
                    gym_env.unwrapped.env.last_u = _state[-1]
                return gym_env.render()  # .transpose(2, 0, 1)

        for _state in states:
            if is_brax and len(_state.q.shape) >= 2:
                _state = jax.tree.map(lambda x: x[0], _state)
            frames.append(render_gym(_state))

        if isinstance(_env.unwrapped, GymnaxBraxWrapper) or not is_brax:
            gym_env.close()
    except Exception as e:
        print(f"Rendering failed with error: {e}")
    return frames
