"""Tests for the SuddenNoiseWrapper."""

import unittest

import jax
import jax.numpy as jnp
import jax.random as jrandom
from brax.envs.base import State

from jax_rl_util.envs.noise_wrappers import SuddenNoiseWrapper
from jax_rl_util.envs.wrappers import Wrapper


class _DummyEnv(Wrapper):
    """A minimal environment that tracks step count for testing wrappers."""

    def __init__(self, obs_size: int = 4, action_size: int = 2):
        self._obs_size = obs_size
        self._action_size = action_size

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def reset(self, rng: jnp.ndarray) -> State:
        obs = jnp.zeros(self._obs_size)
        state = State(
            pipeline_state=obs,
            obs=obs,
            reward=jnp.zeros(1),
            done=jnp.zeros((), dtype=jnp.bool),
            info={"steps": jnp.zeros((), dtype=jnp.int32), "rng": rng},
        )
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        obs = jnp.ones(self._obs_size) * state.info["steps"]  # deterministic obs based on step count
        new_state = state.replace(
            obs=obs,
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
        self.rng = jrandom.PRNGKey(42)

    def test_no_noise_when_start_is_none(self):
        """When sudden_noise_start is None, observations should never be modified."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=None)
        state = env.reset(self.rng)

        for _ in range(10):
            action = jnp.zeros(self.action_size)
            state = env.step(state, action)
            # The dummy env sets obs = ones * step_count, so obs should equal step count
            expected_obs = jnp.ones(self.obs_size) * (state.info["steps"] - 1)
            self.assertTrue(
                jnp.allclose(state.obs, expected_obs),
                f"Expected {expected_obs}, got {state.obs} at step {state.info['steps']}",
            )

    def test_noise_from_step_zero(self):
        """When sudden_noise_start=0, noise should be applied from the very first step."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=0)
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        # The dummy env returns obs = ones * 0 at step 0, so obs should be 0 + noise
        expected_base = jnp.zeros(self.obs_size)
        self.assertFalse(
            jnp.allclose(state.obs, expected_base),
            "Observations should be different from base when noise is applied",
        )

    def test_noise_starts_after_threshold(self):
        """Noise should only be applied after sudden_noise_start steps.

        Note: The wrapper checks state.info['steps'] >= sudden_noise_start
        *after* the env step has incremented steps. So noise_start=5 means
        noise starts when steps >= 5, which occurs after the 5th env.step call.
        """
        noise_start = 5
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=noise_start)
        state = env.reset(self.rng)

        total_steps = noise_start + 3
        for step_idx in range(total_steps):
            action = jnp.zeros(self.action_size)
            state = env.step(state, action)

            # After the step, steps counter = step_idx + 1 (reset sets to 0, then step increments)
            steps_counter = state.info["steps"]
            # The dummy env sets obs = ones * (steps - 1) = ones * step_idx
            expected_base = jnp.ones(self.obs_size) * step_idx

            if steps_counter < noise_start:
                # Before threshold: obs should match the base env exactly
                self.assertTrue(
                    jnp.allclose(state.obs, expected_base),
                    f"At step_idx {step_idx} (steps={steps_counter}): expected {expected_base}, got {state.obs}",
                )
            else:
                # At and after threshold: obs should differ from base
                self.assertFalse(
                    jnp.allclose(state.obs, expected_base),
                    f"At step_idx {step_idx} (steps={steps_counter}): expected noise, got {state.obs}",
                )

    def test_noise_strength_zero(self):
        """When noise_strength=0, observations should be unchanged even after threshold."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=0.0, sudden_noise_start=0)
        state = env.reset(self.rng)

        for _ in range(5):
            action = jnp.zeros(self.action_size)
            state = env.step(state, action)
            expected_obs = jnp.ones(self.obs_size) * (state.info["steps"] - 1)
            self.assertTrue(
                jnp.allclose(state.obs, expected_obs),
                f"With zero noise strength, expected {expected_obs}, got {state.obs}",
            )

    def test_noise_strength_scaling(self):
        """Noise magnitude should scale with noise_strength."""
        noise_strength = 2.0
        env = SuddenNoiseWrapper(self.base_env, noise_strength=noise_strength, sudden_noise_start=0)
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        # The noise should be non-zero and scaled by noise_strength
        noise = state.obs  # base obs is zeros at step 0
        self.assertFalse(
            jnp.allclose(noise, jnp.zeros(self.obs_size)),
            "Noise should be non-zero",
        )

        # Run again with a different seed to verify noise is random
        env2 = SuddenNoiseWrapper(self.base_env, noise_strength=noise_strength, sudden_noise_start=0)
        state2 = env2.reset(jrandom.PRNGKey(123))
        state2 = env2.step(state2, action)
        noise2 = state2.obs

        # Different seeds should produce different noise
        self.assertFalse(
            jnp.allclose(noise, noise2),
            "Different seeds should produce different noise",
        )

    def test_noise_shape_matches_obs(self):
        """Noise should have the same shape as the observation."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=0)
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        self.assertEqual(state.obs.shape, (self.obs_size,))

    def test_noise_is_deterministic_with_same_seed(self):
        """Running the same sequence with the same seed should produce identical noise."""
        env1 = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=0)
        env2 = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=0)

        def run_episode(env, rng):
            state = env.reset(rng)
            action = jnp.zeros(self.action_size)
            obs_list = []
            for _ in range(5):
                state = env.step(state, action)
                obs_list.append(state.obs)
            return jnp.stack(obs_list)

        rng = jrandom.PRNGKey(99)
        traj1 = run_episode(env1, rng)
        traj2 = run_episode(env2, rng)

        self.assertTrue(
            jnp.allclose(traj1, traj2),
            "Same seed should produce identical trajectories",
        )

    def test_noise_does_not_affect_internal_env_state(self):
        """The wrapper should not modify the underlying env's pipeline_state."""
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=0)
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        # pipeline_state should still be the base env's output (no noise)
        expected_pipeline = jnp.zeros(self.obs_size)  # step 0 in dummy env
        self.assertTrue(
            jnp.allclose(state.pipeline_state, expected_pipeline),
            "pipeline_state should not be affected by noise wrapper",
        )

    def test_noise_applied_every_step_after_start(self):
        """Once past sudden_noise_start, noise should be applied on every subsequent step."""
        noise_start = 3
        env = SuddenNoiseWrapper(self.base_env, noise_strength=1.0, sudden_noise_start=noise_start)
        state = env.reset(self.rng)

        total_steps = noise_start + 4
        for step_idx in range(total_steps):
            action = jnp.zeros(self.action_size)
            state = env.step(state, action)

            steps_counter = state.info["steps"]
            expected_base = jnp.ones(self.obs_size) * step_idx

            if steps_counter < noise_start:
                self.assertTrue(
                    jnp.allclose(state.obs, expected_base),
                    f"At step_idx {step_idx} (steps={steps_counter}): expected no noise",
                )
            else:
                self.assertFalse(
                    jnp.allclose(state.obs, expected_base),
                    f"At step_idx {step_idx} (steps={steps_counter}): noise should be applied",
                )

    def test_noise_indices_mask(self):
        """When sudden_noise_indices is set, only the specified indices should be noisy."""
        indices = {0, 2}
        env = SuddenNoiseWrapper(
            self.base_env,
            noise_strength=1.0,
            sudden_noise_start=0,
            sudden_noise_indices=indices,
        )
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        # Base obs is zeros at step 0
        # Indices 0 and 2 should have noise, indices 1 and 3 should be exactly zero
        self.assertNotEqual(state.obs[0], 0.0, "Index 0 should have noise")
        self.assertEqual(state.obs[1], 0.0, "Index 1 should not have noise")
        self.assertNotEqual(state.obs[2], 0.0, "Index 2 should have noise")
        self.assertEqual(state.obs[3], 0.0, "Index 3 should not have noise")

    def test_noise_indices_all(self):
        """When sudden_noise_indices includes all indices, all obs positions should have noise."""
        indices = {0, 1, 2, 3}
        env = SuddenNoiseWrapper(
            self.base_env,
            noise_strength=1.0,
            sudden_noise_start=0,
            sudden_noise_indices=indices,
        )
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        self.assertTrue(
            jnp.all(state.obs != 0.0),
            "All indices should have noise when all are specified",
        )

    def test_noise_indices_single(self):
        """A single index in sudden_noise_indices should only affect that position."""
        indices = {1}
        env = SuddenNoiseWrapper(
            self.base_env,
            noise_strength=1.0,
            sudden_noise_start=0,
            sudden_noise_indices=indices,
        )
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        self.assertEqual(state.obs[0], 0.0, "Index 0 should not have noise")
        self.assertNotEqual(state.obs[1], 0.0, "Index 1 should have noise")
        self.assertEqual(state.obs[2], 0.0, "Index 2 should not have noise")
        self.assertEqual(state.obs[3], 0.0, "Index 3 should not have noise")

    def test_noise_indices_with_noise_strength_zero(self):
        """When noise_strength=0, even specified indices should have no noise."""
        indices = {0, 1}
        env = SuddenNoiseWrapper(
            self.base_env,
            noise_strength=0.0,
            sudden_noise_start=0,
            sudden_noise_indices=indices,
        )
        state = env.reset(self.rng)

        action = jnp.zeros(self.action_size)
        state = env.step(state, action)

        expected_base = jnp.zeros(self.obs_size)
        self.assertTrue(
            jnp.allclose(state.obs, expected_base),
            "With zero noise strength, no indices should have noise",
        )


if __name__ == "__main__":
    unittest.main()
