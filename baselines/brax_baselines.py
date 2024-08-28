"""Train controller using brax' built-in algorithms."""
from dataclasses import dataclass
import functools

from datetime import datetime
import os
from typing import Iterable
from brax.training import acme
import jax
from jax import debug
# import matplotlib.pyplot as plt


from brax import envs
from brax.io import model
# from brax.io import json
# from brax.io import html
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
import numpy as np
from simple_parsing import ArgumentParser

from envs.wrappers import Wrapper
from logging_util import DummyLogger, with_logger
from envs.environments import get_obs_mask

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# if 'COLAB_TPU_ADDR' in os.environ:
#   from jax.tools import colab_tpu
#   colab_tpu.setup_tpu()

DEBUG = False
TRAIN = True


@dataclass
class BraxBaselineParams:
    """Class representing the training parameters for reinforcement learning."""

    env_name: str = 'halfcheetah'
    brax_backend: str = 'spring'
    obs_mask: str | Iterable[int] | None = None
    render: bool = True
    debug: bool = False


# We determined some reasonable hyperparameters offline and share them here.
TRAIN_FNS = {
    'inverted_pendulum': functools.partial(ppo.train, num_timesteps=2_000_000, num_evals=20, reward_scaling=10, episode_length=1000,
                                           normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4,
                                           discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
    'inverted_double_pendulum': functools.partial(ppo.train, num_timesteps=20_000_000, num_evals=20, reward_scaling=10, episode_length=1000,
                                                  normalize_observations=True, action_repeat=1, unroll_length=5, num_minibatches=32,
                                                  num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=2048,
                                                  batch_size=1024, seed=1),
    'ant': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=10, reward_scaling=10, episode_length=1000, normalize_observations=True,
                             action_repeat=1, unroll_length=5, num_minibatches=32, num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
                             entropy_cost=1e-2, num_envs=4096, batch_size=2048, seed=1),
    'humanoid': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=10, reward_scaling=0.1, episode_length=1000, normalize_observations=True,
                                  action_repeat=1, unroll_length=10, num_minibatches=32, num_updates_per_batch=8, discounting=0.97, learning_rate=3e-4,
                                  entropy_cost=1e-3, num_envs=2048, batch_size=1024, seed=1),
    'reacher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True,
                                 action_repeat=4, unroll_length=50, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4,
                                 entropy_cost=1e-3, num_envs=2048, batch_size=256, max_devices_per_host=8, seed=1),
    'humanoidstandup': functools.partial(ppo.train, num_timesteps=100_000_000, num_evals=20, reward_scaling=0.1, episode_length=1000,
                                         normalize_observations=True, action_repeat=1, unroll_length=15, num_minibatches=32, num_updates_per_batch=8,
                                         discounting=0.97, learning_rate=6e-4, entropy_cost=1e-2, num_envs=2048, batch_size=1024, seed=1),
    'hopper': functools.partial(sac.train, num_timesteps=6_553_600, num_evals=20, reward_scaling=30, episode_length=1000, normalize_observations=True,
                                action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=512, grad_updates_per_step=64,
                                max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1),
    'walker2d': functools.partial(sac.train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True,
                                  action_repeat=1, discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=128, grad_updates_per_step=32,
                                  max_devices_per_host=1, max_replay_size=1048576, min_replay_size=8192, seed=1),
    'halfcheetah': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=1, episode_length=1000, normalize_observations=True,
                                     action_repeat=1, unroll_length=20, num_minibatches=32, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4,
                                     entropy_cost=0.001, num_envs=2048, batch_size=512, seed=3),
    'pusher': functools.partial(ppo.train, num_timesteps=50_000_000, num_evals=20, reward_scaling=5, episode_length=1000, normalize_observations=True,
                                action_repeat=1, unroll_length=30, num_minibatches=16, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4,
                                entropy_cost=1e-2, num_envs=2048, batch_size=512, seed=3),
}

MAX_YS = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000,
          'humanoidstandup': 75_000, 'reacher': 5, 'walker2d': 5000, 'pusher': 0}
MIN_YS = {'reacher': -100, 'pusher': -150}


def eval_baseline(params,
                  env_name: str,
                  backend: str,
                  obs_mask=None,
                  steps=1000,
                  render=True,
                  render_start=0,
                  render_steps=1000):
    """Evaluate a baseline model on the given environment."""
    # create an env with auto-reset

    env = envs.get_environment(env_name=env_name, backend=backend)
    if obs_mask is not None:
        env = POBraxWrapper(env, get_obs_mask(env.observation_size, obs_mask))
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)

    if TRAIN_FNS[env_name].func.__module__.split('.')[-2] == 'ppo':
        def make_inference_fn(*args, **kwargs):  # noqa
            return ppo.ppo_networks.make_inference_fn(ppo.ppo_networks.make_ppo_networks(*args, **kwargs))
    elif TRAIN_FNS[env_name].func.__module__.split('.')[-2] == 'sac':
        def make_inference_fn(*args, **kwargs):  # noqa
            return sac.sac_networks.make_inference_fn(sac.sac_networks.make_sac_networks(*args, **kwargs))

    def normalize(x, y): return x  # noqa
    if TRAIN_FNS[env_name].keywords['normalize_observations']:
        normalize = acme.running_statistics.normalize  # noqa

    inference_fn = make_inference_fn(state.obs.shape[-1], env.action_size,
                                     preprocess_observations_fn=normalize)(params)

    jit_inference_fn = jax.jit(inference_fn)

    frames = []
    states = []
    print(f'Running {steps} steps of {env_name} environment')
    for n in range(steps):
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        states.append(state)
        if render and n > render_start and n < render_start + render_steps:
            from brax.io import image
            frames.append(image.render_array(
                env.sys, state.pipeline_state, 256, 256).transpose(2, 0, 1))
            # frames.append(env.render(mode='rgb_array'))

    avg_reward = sum([s.reward for s in states]) / max(sum([s.done for s in states]), 1)
    print(f'average reward: {avg_reward}')
    return avg_reward, frames


class POBraxWrapper(Wrapper, envs.Env):
    """Masks Observations in order to create an POMDP."""

    def __init__(self, env, obs_mask: Iterable[int]):
        """Set obs_mask."""
        super().__init__(env)
        self.obs_mask = obs_mask

    def reset(self, rng):
        """Mask Observation."""
        state = self.env.reset(rng)
        return state.replace(obs=state.obs[..., self.obs_mask])

    def step(self, state, action):
        """Mask Observation."""
        state = self.env.step(state, action)
        return state.replace(obs=state.obs[..., self.obs_mask])

    @property
    def observation_size(self):
        """Get the size of the masked Observation."""
        return len(self.obs_mask)

    @property
    def action_size(self):
        """Get the size of the masked Observation."""
        return self.env.action_size

    @property
    def backend(self):
        """Get the size of the masked Observation."""
        return self.env.backend


def train_brax_baseline(hparams: BraxBaselineParams, logger=DummyLogger()):
    """Train a baseline model for control of a brax physics simulation."""
    env_name = hparams.env_name.replace('brax-', '')
    obs_mask = hparams.obs_mask
    brax_backend = hparams.brax_backend
    env = envs.get_environment(env_name=env_name, backend=brax_backend)

    env = POBraxWrapper(env, get_obs_mask(env.observation_size, obs_mask))
    # state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

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
        ydata.append(metrics['eval/episode_reward'])
        debug.callback(logger.log, metrics, step=num_steps)
        # plt.xlim([0, TRAIN_FNS[env_name].keywords['num_timesteps']])
        # plt.ylim([min_y, max_y])
        # plt.xlabel('# environment steps')
        # plt.ylabel('reward per episode')
        # plt.ylabel('reward per episode')
        # plt.plot(xdata, ydata)
        # plt.savefig(f'/plots/baselines/{env_name}/{num_steps}.png')
        # plt.show()

        def print_progress(num_steps, reward):
            """Print progress async callback."""
            print(f'num_steps: {num_steps}, reward: {reward}')
        debug.callback(print_progress, num_steps, metrics["eval/episode_reward"])

    model_filename = f'artifacts/baselines/{env_name}-{obs_mask}'
    if os.path.exists(model_filename):
        print("Loading existing model")
        params = model.load_params(model_filename)
    else:
        print(f"Starting training for {env_name} environment")
        _, params, _ = TRAIN_FNS[env_name](environment=env, progress_fn=progress)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    model.save_params(model_filename, params)
    # logger.save_model(model_filename)
    params = model.load_params(model_filename)
    avg_reward, frames = eval_baseline(params, env_name, brax_backend, obs_mask)
    logger["eval_reward"] = avg_reward
    if hparams.render and not DEBUG:
        logger.log_video('env/video', np.array(frames), fps=30, caption=f"Reward: {avg_reward:.2f}")


if __name__ == '__main__' and TRAIN:
    parser = ArgumentParser()
    parser.add_arguments(BraxBaselineParams, 'hparams')
    params = parser.parse_args().hparams
    with_logger(train_brax_baseline, params, logger_name="aim", project_name="brax_baselines")
