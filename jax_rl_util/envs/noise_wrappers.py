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
        noise_start: int | None = None,
        rampup_steps: int | None = None,
        noise_indices: set[int] | None = None,
    ):
        super().__init__(env)
        self.noise_strength = noise_strength
        self.sudden_noise_start = noise_start
        self.rampup_steps = rampup_steps
        self.sudden_noise_indices = noise_indices

    def _make_noise(self, rng, noise_shape, strength):
        """Apply noise to a single observation array, optionally masking to specific indices."""
        noise = jrandom.normal(rng, shape=noise_shape) * strength
        if self.sudden_noise_indices is not None:
            mask = (
                jnp.zeros(noise_shape[-1], dtype=bool)
                .at[jnp.array(list(self.sudden_noise_indices))]
                .set(True)
            )
            noise = jnp.where(mask, noise, 0.0)
        return noise

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["noise_global_step"] = state.info.get("noise_global_step", 0)
        return state

    def step(self, state, action: jnp.ndarray, **kwargs):
        global_step = state.info.get("noise_global_step", 0)
        del state.info["noise_global_step"]
        state = self.env.step(state, action, **kwargs)
        if self.sudden_noise_start is not None:
            noise_rng, state.info["rng"] = jrandom.split(state.info["rng"])
            noise_active = global_step >= self.sudden_noise_start
            if self.rampup_steps is not None and self.rampup_steps > 0:
                # Linear ramp up of shift strength over rampup_steps
                rel_step = global_step - self.sudden_noise_start
                strength = self.noise_strength * jnp.clip(
                    rel_step / self.rampup_steps, 0.0, 1.0
                )
            else:
                strength = self.noise_strength
            state = state.replace(
                obs=jax.tree.map(
                    lambda obs: jnp.where(
                        noise_active,
                        obs + self._make_noise(noise_rng, obs.shape, strength),
                        obs,
                    ),
                    state.obs,
                )
            )
        state.info["noise_global_step"] = global_step + 1
        return state


class ShiftWrapper(Wrapper):
    """Adds sudden shifts to the observations after a certain number of steps.

    Can optionally restrict shifts to specific observation indices.
    Supports both flat observations and nested observation trees (e.g. full_obs).
    """

    def __init__(
        self,
        env,
        shift: float,
        shift_start: int | None = None,
        rampup_steps: int | None = None,
        shift_indices: set[int] | None = None,
    ):
        super().__init__(env)
        self.shift_strength = shift
        self.obs_shift_start = shift_start
        self.rampup_steps = rampup_steps
        self.obs_shift_indices = shift_indices

    def _make_shift(self, shift_shape, strength):
        """Apply shift to a single observation array, optionally masking to specific indices."""
        shift = jnp.full(shift_shape, strength)
        if self.obs_shift_indices is not None:
            mask = (
                jnp.zeros(shift_shape[-1], dtype=bool)
                .at[jnp.array(list(self.obs_shift_indices))]
                .set(True)
            )
            shift = jnp.where(mask, shift, 0.0)
        return shift

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["shift_global_step"] = state.info.get("shift_global_step", 0)
        return state

    def step(self, state, action: jnp.ndarray, **kwargs):
        global_step = state.info.get("shift_global_step", 0)
        del state.info["shift_global_step"]
        state = self.env.step(state, action, **kwargs)
        if self.obs_shift_start is not None:
            shift_active = global_step >= self.obs_shift_start
            if self.rampup_steps is not None and self.rampup_steps > 0:
                # Linear ramp up of shift strength over rampup_steps
                rel_step = global_step - self.obs_shift_start
                strength = self.shift_strength * jnp.clip(
                    rel_step / self.rampup_steps, 0.0, 1.0
                )
            else:
                strength = self.shift_strength

            state = state.replace(
                obs=jax.tree.map(
                    lambda obs: jnp.where(
                        shift_active,
                        obs + self._make_shift(obs.shape, strength),
                        obs,
                    ),
                    state.obs,
                )
            )
        state.info["shift_global_step"] = global_step + 1
        return state


class SensorFailureWrapper(Wrapper):
    """Simulates sensor failures by zeroing out specific observation indices after a certain number of steps."""

    def __init__(
        self,
        env,
        failure_start: int | None = None,
        failure_indices: set[int] | None = None,
    ):
        super().__init__(env)
        self.failure_start = failure_start
        self.failure_indices = failure_indices

    def _apply_failure(self, obs):
        """Zero out the specified indices in the observation."""
        if self.failure_indices is not None:
            mask = (
                jnp.zeros(obs.shape[-1], dtype=bool)
                .at[jnp.array(list(self.failure_indices))]
                .set(True)
            )
            obs = jnp.where(mask, 0.0, obs)
        return obs
    
    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["failure_global_step"] = state.info.get("failure_global_step", 0)
        return state

    def step(self, state, action: jnp.ndarray, **kwargs):
        global_step = state.info.get("failure_global_step", 0)
        del state.info["failure_global_step"]
        state = self.env.step(state, action, **kwargs)
        if self.failure_start is not None:
            failure_active = global_step >= self.failure_start
            state = state.replace(
                obs=jax.tree.map(
                    lambda obs: jnp.where(
                        failure_active,
                        self._apply_failure(obs),
                        obs,
                    ),
                    state.obs,
                )
            )
        state.info["failure_global_step"] = global_step + 1
        return state
