from mujoco_playground import State

import jax.numpy as jnp
import jax.random as jrandom

from jax_rl_util.envs.wrappers import Wrapper


class SuddenNoiseWrapper(Wrapper):
    """Adds sudden noise to the observations after a certain number of steps.

    TODO: add tests
    """

    def __init__(
        self, env, noise_strength: float = 1.0, sudden_noise_start: int | None = None
    ):
        super().__init__(env)
        self.noise_strength = noise_strength
        self.sudden_noise_start = sudden_noise_start

    def step(self, state: State, action: jnp.ndarray) -> State:
        state = self.env.step(state, action)
        if (
            self.sudden_noise_start is not None
            and state.info["steps"] >= self.sudden_noise_start
        ):
            noise = (
                jrandom.normal(state.info["rng"], shape=state.obs.shape)
                * self.noise_strength
            )
            state = state.replace(obs=state.obs + noise)
        return state
