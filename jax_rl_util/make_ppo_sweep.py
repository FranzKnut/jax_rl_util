"""Make a wandb grid sweep over recurrent PPO baselines.

Each run trains `jax_rl_util.baselines.ppo_rnn` with `PPOParams` defaults,
overridden by the sweep parameters below. The sweep runs this script as its
program (``--trial``), so create the sweep and start agents from the repo root:

    poetry run python jax_rl_util/make_ppo_sweep.py --dry-run
    poetry run python jax_rl_util/make_ppo_sweep.py
    poetry run wandb agent <entity>/<project>/<sweep_id>
"""

from jax_rl_util.baselines.ppo_rnn import PPOParams, train_and_eval
from jax_rl_util.util.logging_util import with_logger
from jax_rl_util.util.sweeping import create_sweep, is_trial_run, make_sweep_config

PARAMS = {
    # Training
    "debug": {"value": False},
    "seed": {"values": [1, 2, 3, 4, 5]},
    # Logging
    "log_code": {"value": False},
    # Environment
    "env_params": {
        "parameters": {
            "env_name": {
                "values": [
                    "brax-ant",
                    "brax-halfcheetah",
                    "brax-hopper",
                    "brax-reacher",
                    "brax-walker2d",
                    "brax-humanoid",
                ]
            },
            "max_ep_length": {"value": 1000},
            "batch_size": {"value": 32},
            "obs_mask": {"value": None},
        }
    },
    # Optimizer
    "optimizer_params": {"parameters": {
        "learning_rate": {"values": [1e-3]},
        "opt_name": {"value": "adam"},
        "gradient_clip": {"value": None},
    }},
    # RNN
    "num_units": {"value": 32},
    "model": {"values": ["LRU", "CTRNN"]},
    "meta_rl": {"values": [True, False]},
}


def run_trial():
    """Train PPO as one run of the sweep."""
    with_logger(train_and_eval, PPOParams(logging="wandb"))


if __name__ == "__main__":
    if is_trial_run():
        run_trial()
    else:
        name = f"PPO {PARAMS['num_units']['value']}"
        create_sweep(
            make_sweep_config(PARAMS, PPOParams, program=__file__, name=name),
            PPOParams(),
        )
