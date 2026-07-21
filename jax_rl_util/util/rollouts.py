"""Utility for collection rollouts of a given brax environment."""

import os
from dataclasses import dataclass, field
from typing import Literal, Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import numpy as np
import simple_parsing
from jax_rtrl.supervised.example_datasets import load_np_files_from_folder
from jax_rl_util.baselines import load_brax_baseline_inference_fn
from jax_rl_util.envs import EnvironmentConfig, print_env_info
from jax_rl_util.envs.env_util import compute_agg_reward
from jax_rl_util.envs.environments import (
    make_wrapped_env,
    BRAX_ENVS_POS_DIMS,
)
import flax

from jax_rl_util.envs.wrappers import Env


@dataclass
class RolloutConfig:
    """Configuration for collecting rollouts.

    Attributes:
        policy_path (str | None): Path to the policy checkpoint. Defaults to "artifacts/baselines/{env_name}.ckpt".
        ckpt_type (Literal["brax", "orbax"]): Type of checkpoint. Defaults to "brax".
        output_dir (str): Directory to save the rollout data. Defaults to "data".
        env_config (EnvironmentConfig): Configuration for the environment.
        num_rollouts (int): Number of rollouts to collect. Defaults to 100.
        max_steps (int): Maximum number of steps per rollout. Defaults to 1000.
        seed (int): Random seed for reproducibility. Defaults to 0.
    """

    policy_path: str | None = (
        None  # defaults to "jax_rl_util/baselines/trained/{package}/{backend}/{env_name}.ckpt"
    )
    ckpt_type: Literal["brax", "orbax"] = "brax"
    output_dir: str | None = None  # defaults to "data/{package}/{backend}/{env_name}"
    env_config: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            env_name="ant",
            init_kwargs={
                "backend": "mjx",
            },
            # batch_size=1,
            # max_ep_length=100_000,
        )
    )
    num_rollouts: int = 100
    max_steps: int = 1000
    seed: int = 0


@flax.struct.dataclass
class Step:
    act: jnp.ndarray
    obs: jnp.ndarray
    done: jnp.ndarray
    rew: jnp.ndarray

    @property
    def reward(self):
        return self.rew


def make_rollout_fn(
    env: Env, policy_fn: Callable, steps: int, init_carry=None
) -> Callable[[jax.random.PRNGKey, PyTree], tuple[PyTree, jnp.ndarray]]:
    """Make a rollout function for the given environment and policy.

    Parameters
    ----------
    env : Environment
        The environment to rollout in.
    policy_fn : Callable
        The policy function to use for action selection. It should take in observations and return actions.
    steps : int
        The number of steps to rollout.
    init_carry : Any
        The initial carry state for the RNN. If None, the policy is assumed to be feedforward.

    Returns
    -------
    Callable
        A function that takes in a random key and returns the rollout states and actions.
        Shape of the returned states and actions will be (steps, batch_size, ...).
        You can transpose them like this: `jax.tree.map(lambda x: x.swapaxes(0, 1), (states, actions))`
    """
    use_rnn = init_carry is not None

    def _step(carry, _):
        print("Tracing _step")
        prev_state, _hidden, _rng = carry
        _rng, policy_key = jax.random.split(_rng)
        obs = prev_state.obs
        if not getattr(env, "_exclude_current_positions_from_observation", True):
            obs = obs[:, BRAX_ENVS_POS_DIMS[env.env_name] :]
        if use_rnn:
            # Reset when done
            _hidden = jax.tree.map(
                jax.tree_util.Partial(jnp.where, jnp.squeeze(prev_state.done)),
                jax.tree.map(lambda x: x[0], init_carry),
                _hidden,
            )
            _hidden, action = policy_fn(_hidden, obs, policy_key)
        else:
            action = policy_fn(obs, policy_key)
        _state = env.step(prev_state, action)
        return (_state, _hidden, _rng), (prev_state, action)

    def rollout_fn(rng, init_carry):
        reset_key, step_key = jax.random.split(rng)
        env_state = env.reset(reset_key)

        _, (states, actions) = jax.lax.scan(
            _step, (env_state, init_carry, step_key), xs=None, length=steps
        )
        return states, actions

    return rollout_fn


def collect_rollouts(
    config: RolloutConfig, save_rollouts: bool = True, verbose: bool = True
):
    """Collect rollouts for the given environment."""
    rng = jax.random.PRNGKey(config.seed)

    if config.env_config.env_name in BRAX_ENVS_POS_DIMS:
        # We always store the full brax observation here
        config.env_config.init_kwargs["exclude_current_positions_from_observation"] = (
            False
        )

    env, env_info = make_wrapped_env(config.env_config, use_vmap_wrapper=True)
    if verbose:
        print_env_info(env_info)

    if config.ckpt_type == "brax":
        backend = config.env_config.init_kwargs.get("backend")
        policy_fn = load_brax_baseline_inference_fn(
            config.env_config.env_name,
            env.observation_size,
            env.action_size,
            package=env.package_name,
            backend=backend,
        )
        init_carry = None
    elif config.ckpt_type == "orbax":
        from rtr_iil import make_flax_inference_fn  # FIXME

        policy_fn, reset_carry, policy = make_flax_inference_fn(
            config.policy_path, env.observation_size, env.action_size
        )
        rng, policy_key = jax.random.split(rng)
        init_carry = (
            reset_carry(policy_key, (env.observation_size,)) if policy.use_rnn else None
        )

    rollout_fn = make_rollout_fn(env, policy_fn, config.max_steps, init_carry)

    # Make output directory
    if config.output_dir is None:
        output_dir = os.path.join("data", env.package_name)
        if env.package_name == "brax":
            output_dir = os.path.join(output_dir, backend)
        output_dir = os.path.join(output_dir, config.env_config.env_name)
    os.makedirs(output_dir, exist_ok=True)
    total_reward = 0
    total_num_eps = 0
    for i in range(config.num_rollouts):
        rng, _rng = jax.random.split(rng)

        states, actions = rollout_fn(_rng, init_carry)

        states = states.replace(
            done=states.done.at[:, 0].set(0)
        )  # Ensure done is binary (0 or 1) for consistency

        _reward = compute_agg_reward(states.reward, states.done)
        states, actions = jax.tree.map(lambda x: x.swapaxes(0, 1), (states, actions))
        episode_ends = jnp.where(
            jnp.any(states.done[:, 1:], axis=1),
            jnp.array([jnp.where(d, size=1)[0][0] + 1 for d in states.done[:, 1:]]),
            states.done.shape[-1],
        )
        num_episodes = max(1, len(episode_ends))
        print(
            f"Rollout {i:4d}: Collected {num_episodes} episodes. Average reward: {_reward:.2e}. Average Episode length: {jnp.mean(episode_ends):.2e}"
        , end=" ")

        total_reward += _reward
        total_num_eps += num_episodes
        if save_rollouts:
            filename = os.path.join(output_dir, f"rollout-{i}.npz")
            np.savez(
                filename,
                obs=states.obs,
                act=actions,
                rew=states.reward,
                done=states.done,
            )
            print(
                f"Saved to {filename}"
            )
    mean_reward = total_reward / total_num_eps
    print(f"Collected {total_num_eps} episodes. Average reward: {mean_reward}")
    return mean_reward, (states, actions)


def load_rollouts(
    data_folder: str,
    num_files: int | None = None,
    min_ep_length: int | None = None,
    min_reward: float | None = 1000,
):
    """Load rollouts from a given folder. Rollouts are filtered based on the provided criteria.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the rollout files.
    num_files : int | None, optional
        Number of files to load, by default None
    max_ep_length : int | None, optional
        Maximum episode length to consider, by default None
    min_reward : float | None, optional
        Minimum reward to consider, by default None

    Returns
    -------
    see load_np_files_from_folder
    """
    data, file_starts = load_np_files_from_folder(
        data_folder, is_npz=True, num_files=num_files, stack=True
    )
    if data["done"].ndim > 2 and data["done"].shape[1] == 1:
        data = jax.tree.map(
            lambda x: x[:, 0], data
        )  # Remove the extra dimension if present
    data["done"][:, 0] = 0  # Ensure done is binary (0 or 1) for consistency

    if min_ep_length is not None:
        ep_until = jnp.where(
            data["done"].any(axis=1), data["done"].argmax(axis=1), data["done"].shape[1]
        )
        data, file_starts = jax.tree.map(
            lambda x: x[ep_until >= min_ep_length], (data, file_starts)
        )
    if min_reward is not None:
        states = Step(**{k: v for k, v in data.items() if k in Step.__annotations__})
        states = jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), states
        )  # Exclude first step for reward computation
        agg_rewards = compute_agg_reward(states.reward, states.done, agg_fn=None)
        # Filter rollouts based on minimum reward
        data, file_starts = jax.tree.map(
            lambda x: x[agg_rewards >= min_reward], (data, file_starts)
        )
    print(f"Loaded {len(file_starts)} rollouts from {data_folder} after filtering.")
    return data, file_starts


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(RolloutConfig, dest="config")
    args = parser.parse_args()
    avg_reward, _ = collect_rollouts(args.config)
    print(f"Average reward: {avg_reward}")
