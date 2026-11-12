"""Make a wandb grid sweep over the brax baselines.

Each run trains `jax_rl_util.baselines.brax_baselines` with
`BraxBaselineConfig` defaults, overridden by the sweep parameters below. The
algorithm (PPO or SAC) and its hyperparameters come from
`brax_baselines.TRAIN_FNS` for each environment. The sweep runs this script as
its program (``--trial``), so create the sweep and start agents from the repo
root:

    poetry run python jax_rl_util/make_brax_baselines_sweep.py --dry-run
    poetry run python jax_rl_util/make_brax_baselines_sweep.py
    poetry run wandb agent <entity>/<project>/<sweep_id>

The seed is fixed in `TRAIN_FNS`, so there is nothing to gain from sweeping
over seeds here. Trained models are saved per environment and backend under
``baselines/trained``, so runs force re-training instead of loading them.
"""

from jax_rl_util.baselines.brax_baselines import BraxBaselineConfig, train_brax_baseline
from jax_rl_util.util.logging_util import with_logger
from jax_rl_util.util.sweeping import create_sweep, is_trial_run, make_sweep_config

PARAMS = {
    # Training
    "debug": {"value": False},
    "force": {"value": True},
    "num_timesteps": {"value": None},  # None: default of TRAIN_FNS
    "num_evals": {"value": None},  # None: default of TRAIN_FNS
    # Logging
    "log_code": {"value": False},
    "render": {"value": True},
    "eval_steps": {"value": 10_000},
    # Environment
    "env_config": {
        "parameters": {
            "env_name": {
                "values": [
                    "ant",
                    "halfcheetah",
                    "hopper",
                    "walker2d",
                    "humanoid",
                    # "humanoidstandup",
                    # "inverted_pendulum",
                    # "inverted_double_pendulum",
                    # "reacher",
                    # "pusher",
                ]
            },
            "max_ep_length": {"value": 1000},
            "init_kwargs": {
                "parameters": {
                    "backend": {"values": ["spring"]},  # "mjx", "generalized", "positional"
                }
            },
        }
    },
}


def run_trial():
    """Train a brax baseline as one run of the sweep."""
    with_logger(train_brax_baseline, BraxBaselineConfig(logging="wandb"))


if __name__ == "__main__":
    if is_trial_run():
        run_trial()
    else:
        backends = PARAMS["env_config"]["parameters"]["init_kwargs"]["parameters"]["backend"]["values"]
        name = f"brax baselines ({', '.join(backends)})"
        create_sweep(
            make_sweep_config(PARAMS, BraxBaselineConfig, program=__file__, name=name),
            BraxBaselineConfig(),
        )
