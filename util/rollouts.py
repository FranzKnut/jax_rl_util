"""Utility for collection rollouts of a given brax environment."""

import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import simple_parsing
from baselines.brax_baselines import load_brax_model
from envs.environments import EnvironmentConfig, make_env, print_env_info

# HACK
BRAX_BACKEND = "spring"


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

    policy_path: str | None = None  # defaults to "artifacts/baselines/{backend}/{env_name}.ckpt"
    ckpt_type: str = "brax"
    output_dir: str = "data"
    env_config: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            env_name="halfcheetah",
            init_kwargs={
                "backend": BRAX_BACKEND,
                # "exclude_current_positions_from_observation": False,
            },
        )
    )
    num_rollouts: int = 100
    max_steps: int = 1000
    seed: int = 0


def collect_rollouts(config: RolloutConfig, save_rollouts: bool = True, verbose: bool = True):
    """Collect rollouts for the given environment."""
    rng = jax.random.PRNGKey(config.seed)

    env, env_info = make_env(config.env_config, use_vmap_wrapper=False)
    if verbose:
        print_env_info(env_info)

    policy_path = config.policy_path or f"artifacts/baselines/{BRAX_BACKEND}/{config.env_config.env_name}.ckpt"

    if config.ckpt_type == "brax":
        policy_fn = load_brax_model(policy_path, config.env_config.env_name, env.observation_size, env.action_size)
        use_rnn = False
        init_carry = None
    elif config.ckpt_type == "orbax":
        from rtr_iil import make_flax_inference_fn  # FIXME

        policy_fn, policy = make_flax_inference_fn(policy_path, env.observation_size, env.action_size)
        use_rnn = policy.use_rnn
        rng, policy_key = jax.random.split(rng)
        init_carry = policy.initialize_carry(policy_key, (env.observation_size,)) if policy.use_rnn else None

    def _step(carry, _):
        print("Tracing _step")
        prev_state, _hidden, _rng = carry
        _rng, policy_key = jax.random.split(_rng)
        obs = prev_state.obs
        if not getattr(env, "_exclude_current_positions_from_observation", True):
            obs = obs[1:]
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

    output_dir = os.path.join(config.output_dir, config.env_config.env_name)
    os.makedirs(output_dir, exist_ok=True)
    total_reward = 0
    total_num_eps = 0
    for i in range(config.num_rollouts):
        rng, reset_key, step_key = jax.random.split(rng, 3)
        env_state = env.reset(reset_key)

        _, (states, actions) = jax.lax.scan(_step, (env_state, init_carry, step_key), xs=None, length=config.max_steps)
        episode_ends = np.where(states.done)[0] if np.any(states.done) else [len(states.done)]
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
    avg_reward = collect_rollouts(args.config)
    print(f"Average reward: {avg_reward}")
