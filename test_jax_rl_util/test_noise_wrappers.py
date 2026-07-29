"""Tests for the observation-corruption wrappers in noise_wrappers.py."""

import unittest

import jax.numpy as jnp
import jax.random as jrandom
from brax.envs.base import State

from jax_rl_util.envs.noise_wrappers import (
    SensorFailureWrapper,
    ShiftWrapper,
    SuddenNoiseWrapper,
)
from jax_rl_util.envs.wrappers import Wrapper


class _DummyEnv(Wrapper):
    """A minimal environment whose obs deterministically equals the (pre-increment) step count.

    If `tree_obs` is set, obs is a dict of two identical arrays instead of a single array,
    to exercise the wrappers' `jax.tree.map`-based support for nested observations.
    """

    def __init__(self, obs_size: int = 4, action_size: int = 2, tree_obs: bool = False):
        self._obs_size = obs_size
        self._action_size = action_size
        self._tree_obs = tree_obs

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def _make_obs(self, value):
        obs = jnp.ones(self._obs_size) * value
        return {"a": obs, "b": obs} if self._tree_obs else obs

    def reset(self, rng: jnp.ndarray) -> State:
        return State(
            pipeline_state=self._make_obs(0),
            obs=self._make_obs(0),
            reward=jnp.zeros(1),
            done=jnp.zeros((), dtype=jnp.bool),
            info={"steps": jnp.zeros((), dtype=jnp.int32), "rng": rng},
        )

    def step(self, state: State, action: jnp.ndarray) -> State:
        new_state = state.replace(
            obs=self._make_obs(state.info["steps"]),
            reward=jnp.ones(1),
            done=jnp.zeros((), dtype=jnp.bool),
        )
        new_state.info["steps"] = state.info["steps"] + 1
        new_state.info["rng"] = state.info["rng"]
        return new_state


class TestSuddenNoiseWrapper(unittest.TestCase):
    """Tests for SuddenNoiseWrapper."""

    def setUp(self):
        self.obs_size = 4
        self.action_size = 2
        self.base_env = _DummyEnv(obs_size=self.obs_size, action_size=self.action_size)
        self.action = jnp.zeros(self.action_size)
        self.rng = jrandom.PRNGKey(42)

    def test_no_noise_when_start_is_none(self):
        """When noise_start is None, observations should never be modified."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, noise_start=None)
        state = env.reset(self.rng)
        for _ in range(3):
            state = env.step(state, self.action)
            expected_obs = jnp.ones(self.obs_size) * (state.info["steps"] - 1)
            self.assertTrue(jnp.allclose(state.obs, expected_obs))

    def test_noise_threshold_behavior(self):
        """Noise should only be applied once steps >= noise_start (checked post-increment)."""
        noise_start = 3
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, noise_start=noise_start)
        state = env.reset(self.rng)

        for step_idx in range(noise_start + 3):
            state = env.step(state, self.action)
            steps_counter = state.info["steps"]
            expected_base = jnp.ones(self.obs_size) * step_idx
            with self.subTest(step_idx=step_idx):
                if steps_counter <= noise_start:
                    self.assertTrue(jnp.allclose(state.obs, expected_base))
                else:
                    self.assertFalse(jnp.allclose(state.obs, expected_base))

    def test_noise_strength_zero_yields_no_noise(self):
        """With noise_strength=0, obs should be unchanged, with or without an index mask."""
        for indices in (None, {0, 1}):
            with self.subTest(indices=indices):
                env = SuddenNoiseWrapper(
                    self.base_env, noise_strength=0.0, noise_start=0, noise_indices=indices
                )
                state = env.reset(self.rng)
                state = env.step(state, self.action)
                self.assertTrue(jnp.allclose(state.obs, jnp.zeros(self.obs_size)))

    def test_noise_scales_and_is_random(self):
        """Noise should be non-zero, shaped like obs, and differ across seeds."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=2.0, noise_start=0)

        state1 = env.step(env.reset(jrandom.PRNGKey(1)), self.action)
        state2 = env.step(env.reset(jrandom.PRNGKey(2)), self.action)

        self.assertEqual(state1.obs.shape, (self.obs_size,))
        self.assertFalse(jnp.allclose(state1.obs, jnp.zeros(self.obs_size)))
        self.assertFalse(jnp.allclose(state1.obs, state2.obs))

    def test_noise_reproducible_with_same_seed(self):
        """The same seed should produce an identical noise trajectory."""

        def run_episode(rng):
            env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, noise_start=0)
            state = env.reset(rng)
            obs_list = []
            for _ in range(5):
                state = env.step(state, self.action)
                obs_list.append(state.obs)
            return jnp.stack(obs_list)

        rng = jrandom.PRNGKey(99)
        self.assertTrue(jnp.allclose(run_episode(rng), run_episode(rng)))

    def test_noise_indices_mask(self):
        """Only indices in noise_indices should receive noise; others stay exactly at base value."""
        cases = {
            "subset": {0, 2},
            "single": {1},
            "all": {0, 1, 2, 3},
        }
        for name, indices in cases.items():
            with self.subTest(case=name):
                env = SuddenNoiseWrapper(
                    self.base_env, noise_strength=1.0, noise_start=0, noise_indices=indices
                )
                state = env.reset(self.rng)
                state = env.step(state, self.action)
                for i in range(self.obs_size):
                    if i in indices:
                        self.assertNotEqual(state.obs[i], 0.0)
                    else:
                        self.assertEqual(state.obs[i], 0.0)

    def test_noise_rampup_scales_linearly(self):
        """Noise magnitude should ramp from 0 to full strength over rampup_steps."""
        rampup_steps = 4
        env = SuddenNoiseWrapper(
            self.base_env, noise_strength=10.0, noise_start=0, rampup_steps=rampup_steps
        )
        state = env.reset(self.rng)
        state = env.step(state, self.action)  # rel_step=0 -> strength 0
        self.assertTrue(jnp.allclose(state.obs, jnp.zeros(self.obs_size)))

        for _ in range(rampup_steps):
            state = env.step(state, self.action)
        # At and beyond rel_step >= rampup_steps, strength is clipped to full noise_strength.
        self.assertFalse(jnp.allclose(state.obs, jnp.ones(self.obs_size) * state.info["steps"] - 1))

    def test_noise_does_not_affect_pipeline_state(self):
        """The wrapper should not modify the underlying env's pipeline_state."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, noise_start=0)
        state = env.reset(self.rng)
        state = env.step(state, self.action)
        self.assertTrue(jnp.allclose(state.pipeline_state, jnp.zeros(self.obs_size)))


class TestShiftWrapper(unittest.TestCase):
    """Tests for ShiftWrapper."""

    def setUp(self):
        self.obs_size = 4
        self.action_size = 2
        self.base_env = _DummyEnv(obs_size=self.obs_size, action_size=self.action_size)
        self.action = jnp.zeros(self.action_size)
        self.rng = jrandom.PRNGKey(42)

    def test_no_shift_when_start_is_none(self):
        """When shift_start is None, observations should never be modified."""
        env = ShiftWrapper(self.base_env, shift=5.0, shift_start=None)
        state = env.reset(self.rng)
        for _ in range(3):
            state = env.step(state, self.action)
            expected_obs = jnp.ones(self.obs_size) * (state.info["steps"] - 1)
            self.assertTrue(jnp.allclose(state.obs, expected_obs))

    def test_shift_threshold_behavior(self):
        """The constant shift should only be added once steps >= shift_start."""
        shift_start = 3
        shift = 5.0
        env = ShiftWrapper(self.base_env, shift=shift, shift_start=shift_start)
        state = env.reset(self.rng)

        for step_idx in range(shift_start + 3):
            state = env.step(state, self.action)
            steps_counter = state.info["steps"]
            base = jnp.ones(self.obs_size) * step_idx
            with self.subTest(step_idx=step_idx):
                expected = base if steps_counter <= shift_start else base + shift
                self.assertTrue(jnp.allclose(state.obs, expected))

    def test_shift_rampup_scales_linearly(self):
        """Shift magnitude should ramp linearly from 0 to full strength over rampup_steps."""
        rampup_steps = 4
        shift = 8.0
        env = ShiftWrapper(
            self.base_env, shift=shift, shift_start=0, rampup_steps=rampup_steps
        )
        state = env.reset(self.rng)

        for rel_step in range(rampup_steps + 1):
            state = env.step(state, self.action)
            base = jnp.ones(self.obs_size) * rel_step
            expected_strength = shift * min(rel_step / rampup_steps, 1.0)
            with self.subTest(rel_step=rel_step):
                self.assertTrue(jnp.allclose(state.obs, base + expected_strength))

    def test_shift_indices_mask(self):
        """Only indices in shift_indices should be shifted; others stay at base value."""
        indices = {0, 2}
        shift = 3.0
        env = ShiftWrapper(self.base_env, shift=shift, shift_start=0, shift_indices=indices)
        state = env.reset(self.rng)
        state = env.step(state, self.action)

        for i in range(self.obs_size):
            with self.subTest(index=i):
                expected = shift if i in indices else 0.0
                self.assertEqual(state.obs[i], expected)

    def test_shift_supports_tree_obs(self):
        """The shift should apply identically across every leaf of a nested observation tree."""
        tree_env = _DummyEnv(obs_size=self.obs_size, tree_obs=True)
        env = ShiftWrapper(tree_env, shift=3.0, shift_start=0)
        state = env.reset(self.rng)
        state = env.step(state, self.action)

        self.assertTrue(jnp.allclose(state.obs["a"], jnp.full(self.obs_size, 3.0)))
        self.assertTrue(jnp.allclose(state.obs["b"], jnp.full(self.obs_size, 3.0)))

    def test_shift_does_not_affect_pipeline_state(self):
        """The wrapper should not modify the underlying env's pipeline_state."""
        env = ShiftWrapper(self.base_env, shift=5.0, shift_start=0)
        state = env.reset(self.rng)
        state = env.step(state, self.action)
        self.assertTrue(jnp.allclose(state.pipeline_state, jnp.zeros(self.obs_size)))


class TestSensorFailureWrapper(unittest.TestCase):
    """Tests for SensorFailureWrapper."""

    def setUp(self):
        self.obs_size = 4
        self.action_size = 2
        self.base_env = _DummyEnv(obs_size=self.obs_size, action_size=self.action_size)
        self.action = jnp.zeros(self.action_size)
        self.rng = jrandom.PRNGKey(42)

    def test_no_failure_when_start_is_none(self):
        """When failure_start is None, observations should never be modified."""
        env = SensorFailureWrapper(self.base_env, failure_start=None, failure_indices={0, 1})
        state = env.reset(self.rng)
        for _ in range(3):
            state = env.step(state, self.action)
            expected_obs = jnp.ones(self.obs_size) * (state.info["steps"] - 1)
            self.assertTrue(jnp.allclose(state.obs, expected_obs))

    def test_failure_zeroes_indices_after_threshold(self):
        """Indices in failure_indices should read as 0 once steps >= failure_start, not before."""
        failure_start = 3
        indices = {1, 3}
        env = SensorFailureWrapper(
            self.base_env, failure_start=failure_start, failure_indices=indices
        )
        state = env.reset(self.rng)

        for step_idx in range(failure_start + 3):
            state = env.step(state, self.action)
            steps_counter = state.info["steps"]
            base = jnp.ones(self.obs_size) * step_idx
            with self.subTest(step_idx=step_idx):
                if steps_counter <= failure_start:
                    self.assertTrue(jnp.allclose(state.obs, base))
                else:
                    for i in range(self.obs_size):
                        expected = 0.0 if i in indices else step_idx
                        self.assertEqual(state.obs[i], expected)

    def test_failure_without_indices_leaves_obs_unchanged(self):
        """With no failure_indices specified, no observation values should be zeroed."""
        env = SensorFailureWrapper(self.base_env, failure_start=0, failure_indices=None)
        state = env.reset(self.rng)
        for _ in range(3):
            state = env.step(state, self.action)
            expected_obs = jnp.ones(self.obs_size) * (state.info["steps"] - 1)
            self.assertTrue(jnp.allclose(state.obs, expected_obs))

    def test_failure_does_not_affect_pipeline_state(self):
        """The wrapper should not modify the underlying env's pipeline_state."""
        env = SensorFailureWrapper(self.base_env, failure_start=0, failure_indices={0})
        state = env.reset(self.rng)
        state = env.step(state, self.action)
        self.assertTrue(jnp.allclose(state.pipeline_state, jnp.zeros(self.obs_size)))


if __name__ == "__main__":
    unittest.main()
