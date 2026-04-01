"""Integration smoke test for PPO RNN baseline."""

import unittest

import numpy as np

from jax_rl_util.baselines.ppo_rnn import PPOParams, train_and_eval
from jax_rl_util.envs.environments import EnvironmentConfig


class TestPPORNNIntegration(unittest.TestCase):
    def test_train_smoke(self):
        cfg = PPOParams(
            logging=None,
            seed=0,
            episodes=1,
            update_steps=1,
            update_epochs=1,
            collect_steps=2,
            rollout_horizon=2,
            train_batch_size=2,
            eval_every=1,
            eval_steps=2,
            eval_batch_size=1,
            model="GRU",
            num_units=8,
            env_params=EnvironmentConfig(env_name="CartPole-v1", batch_size=2),
        )
        reward = train_and_eval(cfg)
        self.assertTrue(np.isfinite(float(reward)))


if __name__ == "__main__":
    unittest.main()
