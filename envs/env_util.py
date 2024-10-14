from typing import Iterable
import jax
import numpy as np
from jax import numpy as jnp


def render_brax(env, states, render_steps=100, render_start=0, camera=None):
    from brax.io import image

    steps = len(states.pipeline_state.q)
    states_to_render = [
        jax.tree.map(lambda x: x[n], states.pipeline_state)
        for n in range(steps)
        if n > render_start and n < render_start + render_steps
    ]
    camera = camera or ("track" if len(env.sys.cam_bodyid) else -1)
    return image.render_array(env.sys, states_to_render, camera=camera)


def make_obs_mask(base_obs_size: int, obs_mask: Iterable[int] | str | int = None):
    """Get the observation mask from string description.

    obs_mask may take values ['odd', 'even', 'first_half', 'second_half'] or a list of indices.
    """
    # Flat observation size
    if not isinstance(base_obs_size, int):
        base_obs_size = np.prod(base_obs_size)

    if obs_mask == "odd" or obs_mask == "even":
        obs_mask = [i for i in range(base_obs_size) if i % 2 == (obs_mask == "odd")]
    elif obs_mask == "first_half":
        obs_mask = [i for i in range((base_obs_size + 1) // 2)]
    elif obs_mask == "second_half":
        obs_mask = [i for i in range((base_obs_size + 1) // 2, base_obs_size)]
    elif isinstance(obs_mask, int):
        obs_mask = jnp.arange(base_obs_size, dtype=jnp.int32)
    elif obs_mask is None:
        obs_mask = jnp.arange(base_obs_size, dtype=jnp.int32)
    return jnp.array(obs_mask, dtype=jnp.int32)
