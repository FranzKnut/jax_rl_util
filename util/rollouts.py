import os
from baselines.brax_baselines import load_brax_model
import brax.envs.wrappers.gym
import numpy as np
from rtr_iil import make_flax_inference_fn
import jax
from dataclasses import dataclass, field

import simple_parsing

import brax
from jax_rl_util.envs.environments import EnvironmentConfig, make_env, print_env_info


@dataclass
class RolloutConfig:
    policy_path: str = "artifacts/baselines/inverted_pendulum.ckpt"
    ckpt_type: str = "brax"
    output_dir: str = "data"
    env_config: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(env_name="inverted_pendulum", init_kwargs={"backend": "spring"})
    )
    num_rollouts: int = 100
    max_steps: int = 1000
    seed: int = 0


# Collect rollouts for the given environment
def collect_rollouts(config: RolloutConfig):
    rng = jax.random.PRNGKey(config.seed)

    env, env_info = make_env(config.env_config)
    print_env_info(env_info)

    if config.ckpt_type == "brax":
        policy_fn = load_brax_model(
            config.policy_path, config.env_config.env_name, env.observation_size, env.action_size
        )
    elif config.ckpt_type == "orbax":
        policy_fn = make_flax_inference_fn(config.policy_path, env.observation_size, env.action_size)

    def _step(carry, _):
        print("Tracing _step")
        _state, _rng = carry
        _rng, policy_key = jax.random.split(_rng)
        action = policy_fn(_state.obs, policy_key)
        _state = env.step(_state, action)
        return (_state, _rng), (_state, action)

    for i in range(config.num_rollouts):
        rng, reset_key, step_key = jax.random.split(rng, 3)
        env_state = env.reset(reset_key)

        _, (states, actions) = jax.lax.scan(_step, (env_state, step_key), xs=None, length=config.max_steps)
        filename = os.path.join(config.output_dir, config.env_config.env_name, f"rollout-{i}.npz")
        episode_ends = np.where(states.done)[0]
        num_episodes = max(1, len(episode_ends))
        np.savez(
            filename,
            obs=states.obs[: episode_ends[-1]],
            act=actions[: episode_ends[-1]],
            rew=states.reward[: episode_ends[-1]],
            dones=states.done[: episode_ends[-1]],
        )
        print(
            f"Saved {num_episodes} episodes to {filename}. Average reward: {np.sum(states.reward[:episode_ends[-1]]) / num_episodes}"
        )


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(RolloutConfig, dest="config")
    args = parser.parse_args()
    collect_rollouts(args.config)
