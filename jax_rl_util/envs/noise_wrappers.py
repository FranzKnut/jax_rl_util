from mujoco_playground import State

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_rl_util.envs.wrappers import Wrapper


class SuddenNoiseWrapper(Wrapper):
    """Adds sudden noise to the observations after a certain number of steps.

    Can optionally restrict noise to specific observation indices.
    Supports both flat observations and nested observation trees (e.g. full_obs).
    """

    def __init__(
        self,
        env,
        noise_strength: float = 1.0,
        sudden_noise_start: int | None = None,
        sudden_noise_indices: set[int] | None = None,
    ):
        super().__init__(env)
        self.noise_strength = noise_strength
        self.sudden_noise_start = sudden_noise_start
        self.sudden_noise_indices = sudden_noise_indices

    def _make_noise(self, rng, noise_shape):
        """Apply noise to a single observation array, optionally masking to specific indices."""
        noise = jrandom.normal(rng, shape=noise_shape) * self.noise_strength
        if self.sudden_noise_indices is not None:
            mask = (
                jnp.zeros(noise_shape[-1], dtype=bool)
                .at[jnp.array(list(self.sudden_noise_indices))]
                .set(True)
            )
            noise = jnp.where(mask, noise, 0.0)
        return noise
    
    def step(self, state: State, action: jnp.ndarray, noise_global_step: int, **kwargs) -> State:
        state = self.env.step(state, action, **kwargs)
        if self.sudden_noise_start is not None:
            noise_rng, state.info["rng"] = jrandom.split(state.info["rng"])
            noise_active = noise_global_step >= self.sudden_noise_start
            state = state.replace(
                obs=jax.tree.map(
                    lambda obs: jnp.where(
                        noise_active,
                        obs + self._make_noise(noise_rng, obs.shape),
                        obs,
                    ),
                    state.obs,
                )
            )
        return state
