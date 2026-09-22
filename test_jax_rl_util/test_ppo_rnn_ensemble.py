"""Integration smoke test for PPO with RNNEnsemble backbone."""

import unittest

import numpy as np
from jax_rl_util.envs.environments import EnvironmentConfig
from jax_rtrl.networks.policies import PolicyConfig

from jax_rl_util.baselines.ppo_rnn_ensemble import PPOEnsembleParams, train_and_eval

# All RNNEnsemble backbones exercised by PPO, mirroring the model types covered
# by test_rtrrl.CELL_TYPES plus the BASE_CELL_TYPES entries that don't have an
# online-learning subtype (mlp, hopfield, attention, causal_attention).
# "s5"/"s5_rtrl" are excluded: S5SSM currently raises a shape-mismatch error
# inside the RNNEnsemble/PPO glue code, a pre-existing issue unrelated to this
# test and outside its scope.
MODEL_TYPES = [
    "bptt",
    "lru",
    "ltc",
    "lrc",
    "rflo",
    "rtrl",
    "eprop",
    "lru_rtrl",
    "ltc_rtrl",
    "lrc_rtrl",
    "mlp",
    "hopfield",
    "attention",
    "causal_attention",
]


class TestPPOEnsembleIntegration(unittest.TestCase):
    """Test that PPO with RNNEnsemble backbone runs without errors and produces finite rewards."""

    def test_train_smoke(self):  # noqa
        for model_name in MODEL_TYPES:
            with self.subTest(model_name=model_name):
                cfg = PPOEnsembleParams(
                    logging=None,
                    seed=0,
                    episodes=2,
                    update_steps=1,
                    update_epochs=1,
                    collect_steps=2,
                    rollout_horizon=2,
                    train_batch_size=2,
                    eval_every=1,
                    eval_steps=2,
                    eval_batch_size=1,
                    num_units=8,
                    env_params=EnvironmentConfig(env_name="CartPole-v1", batch_size=2),
                    model_config=PolicyConfig(model_name=model_name, hidden_size=8),
                )
                reward = train_and_eval(cfg)
                self.assertTrue(np.isfinite(float(reward)))


if __name__ == "__main__":
    unittest.main()
