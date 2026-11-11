"""Make a wandb grid sweep over PPO with an RNN ensemble backbone.

Each run trains `jax_rl_util.baselines.ppo_rnn_ensemble` with
`PPOEnsembleParams` defaults, overridden by the sweep parameters below. The
sweep runs this script as its program (``--trial``), so create the sweep and
start agents from the repo root:

    poetry run python jax_rl_util/make_ppo_rnn_ensemble_sweep.py --dry-run
    poetry run python jax_rl_util/make_ppo_rnn_ensemble_sweep.py
    poetry run wandb agent <entity>/<project>/<sweep_id>

Online models such as ``rflo`` or ``lru_rtrl`` are trained with their BPTT
counterparts, see `ppo_rnn_ensemble.MODEL_DOWNCAST`.
"""

from jax_rl_util.baselines.ppo_rnn_ensemble import PPOEnsembleParams, train_and_eval
from jax_rl_util.util.logging_util import with_logger
from jax_rl_util.util.sweeping import create_sweep, is_trial_run, make_sweep_config

PARAMS = {
    # Training
    "debug": {"value": False},
    "seed": {"values": [1, 2, 3]},
    # Logging
    "log_code": {"value": False},
    "save_model": {"value": False},
    # Environment
    "env_params": {
        "parameters": {
            "env_name": {"values": ["hopper", "halfcheetah", "ant"]},
            "batch_size": {"value": 32},
            "max_ep_length": {"value": 1000},
            "obs_mask": {"value": None},
        }
    },
    # Optimizer
    "optimizer_params": {
        "parameters": {
            "opt_name": {"value": "adam"},
            "learning_rate": {"values": [1e-3]},
        }
    },
    # RNN ensemble
    "train_encoder": {"value": False},
    "model_config": {
        "parameters": {
            "model_name": {"values": ["bptt", "lru", "lrc"]},
            "hidden_size": {"value": 32},
            "num_layers": {"value": 1},
            "num_modules": {"values": [1, 5]},
        }
    },
}


def run_trial():
    """Train PPO with an RNN ensemble as one run of the sweep."""
    with_logger(train_and_eval, PPOEnsembleParams(logging="wandb"))


if __name__ == "__main__":
    if is_trial_run():
        run_trial()
    else:
        modules = PARAMS["model_config"]["parameters"]["num_modules"]["values"]
        name = f"PPO ensemble {', '.join(map(str, modules))} modules"
        create_sweep(
            make_sweep_config(PARAMS, PPOEnsembleParams, program=__file__, name=name),
            PPOEnsembleParams(),
        )
