import os
from baselines.brax_baselines import load_brax_model
import brax.envs.wrappers.gym
import numpy as np
from sqlalchemy import all_
from rtr_iil import make_flax_inference_fn
import jax
from dataclasses import dataclass, field

import simple_parsing

import brax
from jax_rl_util.envs.environments import EnvironmentConfig, make_env, print_env_info


@dataclass
class RolloutConfig:
    """Configuration for collecting rollouts.

    Attributes:
        policy_path (str | None): Path to the policy checkpoint. Defaults to "artifacts/baselines/{env_name}.ckpt".
        ckpt_type (str): Type of checkpoint. Defaults to "brax".
        output_dir (str): Directory to save the rollout data. Defaults to "data".
        env_config (EnvironmentConfig): Configuration for the environment. Defaults to an EnvironmentConfig with
                                        env_name="inverted_pendulum" and init_kwargs={"backend": "spring"}.
        num_rollouts (int): Number of rollouts to collect. Defaults to 100.
        max_steps (int): Maximum number of steps per rollout. Defaults to 1000.
        seed (int): Random seed for reproducibility. Defaults to 0.
    """

    policy_path: str | None = None  # defaults to "artifacts/baselines/{env_name}.ckpt"
    ckpt_type: str = "brax"
    output_dir: str = "data"
    env_config: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(env_name="inverted_pendulum", init_kwargs={"backend": "spring"})
    )
    num_rollouts: int = 100
    max_steps: int = 1000
    seed: int = 0


# Collect rollouts for the given environment
def collect_rollouts(config: RolloutConfig, save_rollouts: bool = True):
    rng = jax.random.PRNGKey(config.seed)

    env, env_info = make_env(config.env_config)
    print_env_info(env_info)

    policy_path = config.policy_path or f"artifacts/baselines/{config.env_config.env_name}.ckpt"

    if config.ckpt_type == "brax":
        policy_fn = load_brax_model(policy_path, config.env_config.env_name, env.observation_size, env.action_size)
    elif config.ckpt_type == "orbax":
        policy_fn = make_flax_inference_fn(policy_path, env.observation_size, env.action_size)
    avg_reward = collect_rollouts(env, policy_fn, config.seed, config.output_dir, config.num_rollouts, config.max_steps)
    print(f"Average reward: {avg_reward}")
    rng = jax.random.PRNGKey(config.seed)

    def _step(carry, _):
        print("Tracing _step")
        _state, _rng = carry
        _rng, policy_key = jax.random.split(_rng)
        action = policy_fn(_state.obs, policy_key)
        _state = env.step(_state, action)
        return (_state, _rng), (_state, action)

    output_dir = os.path.join(config.output_dir, config.env_config.env_name)
    os.makedirs(output_dir, exist_ok=True)
    total_reward = 0
    total_num_eps = 0
    for i in range(config.num_rollouts):
        rng, reset_key, step_key = jax.random.split(rng, 3)
        env_state = env.reset(reset_key)

        _, (states, actions) = jax.lax.scan(_step, (env_state, step_key), xs=None, length=config.max_steps)
        episode_ends = np.where(states.done)[0]
        num_episodes = max(1, len(episode_ends))
        total_reward += np.sum(states.reward[: episode_ends[-1]])
        total_num_eps += num_episodes
        if save_rollouts:
            filename = os.path.join(output_dir, f"rollout-{i}.npz")
            np.savez(
                filename,
                obs=states.obs[: episode_ends[-1]],
                act=actions[: episode_ends[-1]],
                rew=states.reward[: episode_ends[-1]],
                done=states.done[: episode_ends[-1]],
            )
            print(
                f"Saved {num_episodes} episodes to {filename}. Average reward: {np.sum(states.reward[:episode_ends[-1]]) / num_episodes}"
            )
    return total_reward / total_num_eps


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(RolloutConfig, dest="config")
    args = parser.parse_args()
    collect_rollouts(args.config)
