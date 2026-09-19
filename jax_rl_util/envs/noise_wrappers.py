from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_rl_util.envs.wrappers import Wrapper


def _mask_to_indices(values, indices):
    """Zero out every entry whose last-axis index is not in `indices`."""
    mask = jnp.zeros(values.shape[-1], dtype=bool).at[jnp.array(list(indices))].set(True)
    return jnp.where(mask, values, 0.0)


class SuddenNoiseWrapper(Wrapper):
    """Adds sudden noise to the observations and/or actions after a certain number of steps.

    `noise_target` selects where the noise goes. Noise on the action is applied
    before the environment step, noise on the observation after it. Either side
    can be restricted to specific indices; `None` means all indices.
    Supports both flat observations and nested observation trees (e.g. full_obs).
    """

    def __init__(
        self,
        env,
        noise_strength: float = 1.0,
        noise_start: int | None = None,
        rampup_steps: int | None = None,
        noise_target: Literal["obs", "act", "both"] = "obs",
        obs_noise_indices: set[int] | None = None,
        act_noise_indices: set[int] | None = None,
    ):
        super().__init__(env)
        self.noise_strength = noise_strength
        self.sudden_noise_start = noise_start
        self.rampup_steps = rampup_steps
        self.noise_target = noise_target
        self.obs_noise_indices = obs_noise_indices
        self.act_noise_indices = act_noise_indices

    def _strength(self, global_step):
        """Noise strength at this step, linearly ramped up if configured."""
        if self.rampup_steps is not None and self.rampup_steps > 0:
            rel_step = global_step - self.sudden_noise_start
            return self.noise_strength * jnp.clip(rel_step / self.rampup_steps, 0.0, 1.0)
        return self.noise_strength

    def _make_obs_noise(self, rng, noise_shape, strength):
        """Sample observation noise, masked to `obs_noise_indices` if given."""
        noise = jrandom.normal(rng, shape=noise_shape) * strength
        if self.obs_noise_indices is not None:
            noise = _mask_to_indices(noise, self.obs_noise_indices)
        return noise

    def _make_act_noise(self, rng, noise_shape, strength):
        """Sample action noise, masked to `act_noise_indices` if given."""
        noise = jrandom.normal(rng, shape=noise_shape) * strength
        if self.act_noise_indices is not None:
            noise = _mask_to_indices(noise, self.act_noise_indices)
        return noise

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["noise_global_step"] = state.info.get("noise_global_step", 0)
        return state

    def step(self, state, action: jnp.ndarray, **kwargs):
        global_step = state.info.get("noise_global_step", 0)
        del state.info["noise_global_step"]
        noise_active = (
            global_step >= self.sudden_noise_start
            if self.sudden_noise_start is not None
            else False
        )

        if self.sudden_noise_start is not None and self.noise_target in ("act", "both"):
            act_rng, state.info["rng"] = jrandom.split(state.info["rng"])
            action = jnp.where(
                noise_active,
                action
                + self._make_act_noise(act_rng, action.shape, self._strength(global_step)),
                action,
            )

        state = self.env.step(state, action, **kwargs)
        if self.sudden_noise_start is not None and self.noise_target in ("obs", "both"):
            noise_rng, state.info["rng"] = jrandom.split(state.info["rng"])
            strength = self._strength(global_step)
            state = state.replace(
                obs=jax.tree.map(
                    lambda obs: jnp.where(
                        noise_active,
                        obs + self._make_obs_noise(noise_rng, obs.shape, strength),
                        obs,
                    ),
                    state.obs,
                )
            )
        state.info["noise_global_step"] = global_step + 1
        return state


class ShiftWrapper(Wrapper):
    """Adds sudden shifts to the observations and/or actions after a certain number of steps.

    `shift_target` selects where the shift goes. The action shift is applied
    before the environment step, the observation shift after it. Either side can
    be restricted to specific indices; `None` means all indices.
    Supports both flat observations and nested observation trees (e.g. full_obs).
    """

    def __init__(
        self,
        env,
        shift: float,
        shift_start: int | None = None,
        rampup_steps: int | None = None,
        shift_target: Literal["obs", "act", "both"] = "obs",
        obs_shift_indices: set[int] | None = None,
        act_shift_indices: set[int] | None = None,
    ):
        super().__init__(env)
        self.shift_strength = shift
        self.shift_start = shift_start
        self.rampup_steps = rampup_steps
        self.shift_target = shift_target
        self.obs_shift_indices = obs_shift_indices
        self.act_shift_indices = act_shift_indices

    def _strength(self, global_step):
        """Shift strength at this step, linearly ramped up if configured."""
        if self.rampup_steps is not None and self.rampup_steps > 0:
            rel_step = global_step - self.shift_start
            return self.shift_strength * jnp.clip(rel_step / self.rampup_steps, 0.0, 1.0)
        return self.shift_strength

    def _make_obs_shift(self, shift_shape, strength):
        """Constant observation shift, masked to `obs_shift_indices` if given."""
        shift = jnp.full(shift_shape, strength)
        if self.obs_shift_indices is not None:
            shift = _mask_to_indices(shift, self.obs_shift_indices)
        return shift

    def _make_act_shift(self, shift_shape, strength):
        """Constant action shift, masked to `act_shift_indices` if given."""
        shift = jnp.full(shift_shape, strength)
        if self.act_shift_indices is not None:
            shift = _mask_to_indices(shift, self.act_shift_indices)
        return shift

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["shift_global_step"] = state.info.get("shift_global_step", 0)
        return state

    def step(self, state, action: jnp.ndarray, **kwargs):
        global_step = state.info.get("shift_global_step", 0)
        del state.info["shift_global_step"]
        shift_active = (
            global_step >= self.shift_start if self.shift_start is not None else False
        )

        if self.shift_start is not None and self.shift_target in ("act", "both"):
            action = jnp.where(
                shift_active,
                action + self._make_act_shift(action.shape, self._strength(global_step)),
                action,
            )

        state = self.env.step(state, action, **kwargs)
        if self.shift_start is not None and self.shift_target in ("obs", "both"):
            strength = self._strength(global_step)
            state = state.replace(
                obs=jax.tree.map(
                    lambda obs: jnp.where(
                        shift_active,
                        obs + self._make_obs_shift(obs.shape, strength),
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
