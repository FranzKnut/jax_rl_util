from collections.abc import Iterable
from typing import NamedTuple
from functools import partial
import jax
from jax import numpy as jnp
from jaxtyping import PyTree


class Transition(NamedTuple):
    """A transition used in batch updates."""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    state: jnp.ndarray


def calculate_gae(
    transitions: Transition, values, last_val, gamma=0.99, gae_lambda=0.95
):
    """Compute the generalized advantage estimates."""

    def _get_advantages(carry, _batch: tuple[Transition, jax.Array]):
        gae, next_value = carry
        _transition, _value = _batch
        _done, _reward = _transition.done, _transition.reward.squeeze()
        delta = _reward + gamma * next_value * (1 - _done) - _value
        gae = delta + gamma * gae_lambda * (1 - _done) * gae
        return (gae, _value), gae

    _, advantages = jax.lax.scan(
        jax.vmap(_get_advantages),
        (jnp.zeros_like(last_val), last_val),
        (transitions, values),
        reverse=True,
        # unroll=rollout_horizon,
    )
    return advantages, advantages + transitions.value
