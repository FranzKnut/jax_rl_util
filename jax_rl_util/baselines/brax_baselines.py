"""Train controller using brax' built-in algorithms."""

import functools
import os
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint

import jax
import numpy as np
import simple_parsing
from brax.io import model
from brax.training import acme
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from jax import debug
from jax import numpy as jnp
from jax_rl_util.envs.environments import EnvironmentConfig, get_env, make_wrapped_env
from jax_rl_util.util.logging_util import DummyLogger, LoggableConfig, with_logger

from jax_rl_util.envs.env_util import render_frames

DEBUG = False
TRAIN = True


os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"

# try_init_metal()


@dataclass
class BraxBaselineConfig(LoggableConfig):
    """Class representing the training parameters for reinforcement learning."""

    project_name: str = "brax_baselines"
    logging: str | None = "wandb"
    force: bool = False
    env_config: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(env_name="PandaPickCubeOrientation")
    )
    render: bool = True


ppo_defaults = {
    "num_timesteps": 1_000_000,
    "episode_length": 1000,
}

# We determined some reasonable hyperparameters offline and share them here.
TRAIN_FNS = {
    "inverted_pendulum": functools.partial(
        ppo.train,
        num_timesteps=2_000_000,
        num_evals=20,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=1024,
        seed=1,
    ),
    "inverted_double_pendulum": functools.partial(
        ppo.train,
        num_timesteps=20_000_000,
        num_evals=20,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=1024,
        seed=1,
    ),
    "ant": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=5,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=4096,
        batch_size=2048,
        seed=1,
    ),
    "humanoid": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        seed=1,
    ),
    "reacher": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=20,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=4,
        unroll_length=50,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=256,
        max_devices_per_host=8,
        seed=1,
    ),
    "humanoidstandup": functools.partial(
        ppo.train,
        num_timesteps=100_000_000,
        num_evals=20,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=15,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=6e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=1024,
        seed=1,
    ),
    "hopper": functools.partial(
        sac.train,
        num_timesteps=6_553_600,
        num_evals=20,
        reward_scaling=30,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=512,
        grad_updates_per_step=64,
        max_devices_per_host=1,
        max_replay_size=1048576,
        min_replay_size=8192,
        seed=1,
    ),
    "walker2d": functools.partial(
        sac.train,
        num_timesteps=7_864_320,
        num_evals=20,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=128,
        batch_size=128,
        grad_updates_per_step=32,
        max_devices_per_host=1,
        max_replay_size=1048576,
        min_replay_size=8192,
        seed=1,
    ),
    "halfcheetah": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=20,
        reward_scaling=1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=2048,
        batch_size=512,
        seed=3,
    ),
    "pusher": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=20,
        reward_scaling=5,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=30,
        num_minibatches=16,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=2048,
        batch_size=512,
        seed=3,
    ),
    "hebi": functools.partial(
        ppo.train,
        num_timesteps=50_000_000,
        num_evals=10,
        reward_scaling=10,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.97,
        learning_rate=3e-4,
        num_envs=1024,
        batch_size=512,
        seed=1,
        unroll_length=5,
    ),
}

MAX_YS = {
    "ant": 8000,
    "halfcheetah": 8000,
    "hopper": 2500,
    "humanoid": 13000,
    "humanoidstandup": 75_000,
    "reacher": 5,
    "walker2d": 5000,
    "pusher": 0,
}
MIN_YS = {"reacher": -100, "pusher": -150}


def load_brax_model(path, env_name: str, obs_size: int, act_size: int):
    """Load a trained model from given path."""
    params = model.load_params(path)

    def normalize(x, y):
        return x  # noqa

    _train_fn = TRAIN_FNS.get(env_name, functools.partial(ppo.train, **ppo_defaults))

    if _train_fn.keywords.get("normalize_observations", False):
        normalize = acme.running_statistics.normalize  # noqa

    if _train_fn.func.__module__.split(".")[-2] == "ppo":

        def make_inference_fn(*args, **kwargs):  # noqa
            return ppo.ppo_networks.make_inference_fn(
                ppo.ppo_networks.make_ppo_networks(*args, **kwargs)
            )
    elif _train_fn.func.__module__.split(".")[-2] == "sac":

        def make_inference_fn(*args, **kwargs):  # noqa
            return sac.sac_networks.make_inference_fn(
                sac.sac_networks.make_sac_networks(*args, **kwargs)
            )

    _fn = make_inference_fn(obs_size, act_size, preprocess_observations_fn=normalize)(
        params
    )
    return jax.jit(lambda obs, key: _fn(obs, key)[0])


# @jax.jit # TODO: refactor to make_eval_fn() -> jitted function
def eval_baseline(
    env_config,
    path: str,
    steps=10000,
):
    """Evaluate a baseline model on the given environment."""
    # create an env with auto-reset
    env = make_wrapped_env(env_config)[0]
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(seed=rng)
    jit_inference_fn = load_brax_model(
        path, env_config.env_name, env.observation_size, env.action_size
    )

    print(f"Running {steps} steps of {env_config.env_name} environment")

    def eval_step(carry, n):
        state, rng = carry
        act_rng, rng = jax.random.split(rng)
        act = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        return (state, rng), state

    _, states = jax.lax.scan(eval_step, (state, rng), jnp.arange(steps))
    avg_reward = jnp.sum(states.reward) / jnp.maximum(jnp.sum(states.done), 1)
    return avg_reward, states


def train_brax_baseline(config: BraxBaselineConfig, logger=DummyLogger()):
    """Train a baseline model for control of a brax physics simulation."""
    env_name = config.env_config.env_name
    env = get_env(config.env_config)

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps: int, metrics: dict):
        """Log progress.

        Args:
            num_steps (int): number of steps so far
            metrics (dict): metrics to log

        """
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        debug.callback(logger.log, metrics, step=num_steps)

        def print_progress(step, rew):
            print(f"num_steps: {step:.2f}, reward: {rew}")

        debug.callback(print_progress, num_steps, metrics["eval/episode_reward"])

    file_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = file_dir + f"/trained/brax_baselines/{env.package_name}"
    if env.package_name == "brax":
        model_filename += (
            f"/{config.env_config.env_kwargs.get('backend', 'generalized')}"
        )
    model_filename += f"/{env_name}.ckpt"

    wrap_env_fn = None
    ppo_config = ppo_defaults.copy()
    if env.package_name == "mujoco_playground":
        import mujoco_playground
        from mujoco_playground import wrapper

        wrap_env_fn = wrapper.wrap_for_brax_training
        _env_name = env_name.replace("mujoco_playground", "")
        if _env_name in mujoco_playground.manipulation._envs:
            from mujoco_playground.config import manipulation_params as default_params
        elif _env_name in mujoco_playground.locomotion._envs:
            from mujoco_playground.config import locomotion_params as default_params
        elif _env_name in mujoco_playground.dm_control_suite._envs:
            from mujoco_playground.config import (
                dm_control_suite_params as default_params,
            )
        else:
            print(
                f"Env {_env_name} not found in mujoco_playground registry. Using default PPO params."
            )
        # TODO: allow sac training for mujoco_playground envs
        ppo_config = default_params.brax_ppo_config(env_name)
        network_factory = ppo.ppo_networks.make_ppo_networks
        if "network_factory" in ppo_config:
            # HACK: construct network factory from mujoco_playground params
            network_factory = functools.partial(
                network_factory, **ppo_config.network_factory
            )
            del ppo_config["network_factory"]
    pprint(ppo_config)
    _train_fn = TRAIN_FNS.get(
        env_name, functools.partial(ppo.train, **dict(ppo_config))
    )

    if os.path.exists(model_filename) and not config.force:
        print("Loading existing model")
        params = model.load_params(model_filename)
    else:
        print(f"Starting training for {env_name} environment")
        _, params, _ = _train_fn(
            environment=env,
            progress_fn=progress,
            wrap_env_fn=wrap_env_fn,
            network_factory=network_factory,
        )
        print(f"time to jit: {times[1] - times[0]}")
        print(f"time to train: {times[-1] - times[1]}")

    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    model.save_params(model_filename, params)
    # logger.save_model(model_filename)
    print(f"Saved model to {model_filename}")
    avg_reward, states = eval_baseline(config.env_config, model_filename)
    print(f"average reward: {avg_reward}")
    if config.render and not DEBUG:
        print("Rendering...")
        frames = render_frames(env, states, start_idx=0, end_idx=200)
        logger.log_video(
            "env/video",
            np.array(frames),
            fps=30,
            caption=f"Reward: {avg_reward.mean():.2f}",
        )
    logger["eval_reward"] = avg_reward


if __name__ == "__main__" and TRAIN:
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(BraxBaselineConfig, "params")
    config: BraxBaselineConfig = parser.parse_args().params

    with_logger(train_brax_baseline, config)
