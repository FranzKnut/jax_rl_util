""" Meta-RL with Gymnax

Adapted from https://github.com/RobertTLange/gymnax/blob/main/examples/03_meta_a2c.ipynb
Author: Julian Lemmel
"""
import os
from typing import Tuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import gymnax
import pandas as pd
from experiments.wandb_sweeps.AllGymnax import all_env_names

jax.config.update("jax_debug_nans", True)

NUM_STEPS = 100
NUM_RUNS = 5
NUM_UNITS = 32
BATCH_SIZE = 32
LR = 1e-2
OUTPUT_PATH = './logs/gymnax_metarl/'
ENV_NAMES = all_env_names
# ENV_NAMES = ['Catch-bsuite']


class LSTMMetaRL(hk.RNNCore):
    """Simple LSTM Wrapper with flexible output head."""

    def __init__(
            self,
            num_hidden_units: int = 32,
            num_output_units: int = 2) -> None:
        super().__init__(name="LSTMMetaRL")
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        self._lstm_core = hk.LSTM(self.num_hidden_units)
        self._policy_head = hk.Linear(self.num_output_units)
        self._value_head = hk.Linear(1)

    def initial_state(self, batch_size: int) -> hk.LSTMState:
        return self._lstm_core.initial_state(batch_size)

    def __call__(
            self,
            x: chex.Array,
            state: hk.LSTMState) -> Tuple[hk.LSTMState, chex.Array, chex.Array]:
        output, next_state = self._lstm_core(x, state)
        policy_logits = self._policy_head(output)
        value = self._value_head(output)

        return next_state, policy_logits, value


def compute_returns(rewards, gamma=1.):
    """Compute list of returns up to T."""
    R = 0
    returns = jnp.zeros(rewards.shape[0])
    counter = rewards.shape[0] - 1
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns = returns.at[counter].set(R)
        counter -= 1
    return returns


def rollout(rng_input, gamma, env_params, steps_in_episode):
    """Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, env_state = env.reset(rng_reset, env_params)
    obs = jnp.ravel(obs)

    lstm = LSTMMetaRL(NUM_UNITS, 2)
    net_state = lstm.initial_state(None)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, env_state, net_state, rng = state_input
        rng, rng_step, rng_act = jax.random.split(rng, 3)
        net_state, logits, value = lstm(obs, net_state)
        policy = distrax.Categorical(logits=logits)
        entropy = policy.entropy()
        action, log_prob = policy.sample_and_log_prob(seed=rng_act, sample_shape=())
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        carry = [jnp.ravel(next_obs), next_env_state, net_state, rng]
        return carry, [log_prob, value, entropy, reward, done]

    # Scan over episode step loop
    _, scan_out = hk.scan(
        policy_step,
        [obs, env_state, net_state, rng_episode],
        (),
        steps_in_episode
    )
    # Unpack episode statistics
    log_probs, values, entropies, rewards, done = scan_out
    returns = compute_returns(rewards, gamma)
    advantages = returns - values
    return log_probs, advantages, entropies, jnp.sum(rewards), jnp.sum(done)


# Batch rollout helper - vmap over rngs
rollout_fn = hk.without_apply_rng(hk.transform(rollout))
batch_rollout_fn = jax.vmap(rollout_fn.apply, in_axes=(None, 0, None, None, None))


def a2c_loss(log_probs, advantages, entropies, entropy_coeff, value_coeff):
    """Compute the actor-critic REINFORCE loss with entropy regularization."""
    # ============ YOUR CODE HERE =============
    actor_loss = -(log_probs * jax.lax.stop_gradient(advantages)).mean()
    critic_loss = (advantages**2).mean()
    entropy_loss = entropies.mean()
    ac_loss = actor_loss + value_coeff * critic_loss - entropy_coeff * entropy_loss
    return ac_loss


@jax.jit
def batch_a2c_loss(net_params, rng_batch, gamma, entropy_coeff, value_coeff, env_params):
    """Batch forward pass for multiple actors & compute average A2C loss."""
    batch_loss_fn = jax.vmap(a2c_loss, in_axes=(0, 0, 0, None, None))
    log_probs, advantages, entropies, rewards, dones = batch_rollout_fn(net_params, rng_batch, gamma, env_params, NUM_STEPS)
    batch_loss = batch_loss_fn(log_probs, advantages, entropies, entropy_coeff, value_coeff)
    return batch_loss.mean(), jnp.sum(rewards) / jnp.clip(jnp.sum(dones), a_min=1)


def run_a2c_training(rng, env_params, verbose):
    num_updates = 3000
    num_workers = BATCH_SIZE
    beta_ent = 0.5
    beta_ent_decay = 0.99
    beta_value = 0.1
    gamma = 0.99
    track_loss, track_rewards = [], []

    net_params = rollout_fn.init(rng, rng, gamma, env_params, NUM_STEPS)
    optimizer = optax.chain(
        # optax.clip_by_global_norm(5.0),
        optax.scale_by_adam(),
        optax.scale(-LR),
    )
    opt_state = optimizer.init(net_params)

    for up in range(num_updates):
        # Split random number keys for episode rollout
        rng, rng_b = jax.random.split(rng)
        rng_batch = jax.random.split(rng_b, num_workers)

        # ============ YOUR CODE HERE =============
        # Rollout batch episodes for workers via vmap & perform gradient update
        out, grads = jax.value_and_grad(batch_a2c_loss, has_aux=True)(net_params, rng_batch, gamma, beta_ent, beta_value, env_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        net_params = optax.apply_updates(net_params, updates)

        # Decay the exploration loss coefficient - meta-exploration
        beta_ent = jnp.clip(beta_ent * beta_ent_decay, 0.01, 1.0)

        # Track loss and print performance
        track_loss.append(out[0])
        track_rewards.append(out[1])
        # track_regrets.append(out[1][1])
        if verbose and up % 200 == 0:
            print(f"# Updates: {up} | Loss: {out[0]} | Return {out[1]}  | b_e: {beta_ent}")
    return net_params, track_loss, track_rewards  # , track_regrets


experiments = pd.DataFrame(index=[f"run-{i}" for i in range(NUM_RUNS)])
for name in ENV_NAMES:
    # Rollout an episode and return collected stats
    # for length in [4, 8, 16, 32, 64]:
    all_rewards = []
    for i in range(NUM_RUNS):
        print(f"Running {name} run {i+1}/{NUM_RUNS}")
        env, env_params = gymnax.make(name)
        # env_params = env.default_params.replace(memory_length=length)
        rng = jax.random.PRNGKey(i)
        net_params_10, loss_10, rewards_10 = run_a2c_training(rng, env_params, verbose=True)
        all_rewards.append(rewards_10[-1])
    experiments[name] = all_rewards
    experiments.to_csv(os.path.join(OUTPUT_PATH, 'AllGymnax.csv'))
