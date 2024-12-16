"""PPO implementation in JAX."""

import functools
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Sequence

import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import simple_parsing
import tensorflow_probability.substrates.jax as tfp
from brax.training.acme import running_statistics
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt

from jax_rl_util.envs.environments import EnvironmentConfig, make_env, print_env_info
from jax_rl_util.envs.plot_drones import plot_from_file
from jax_rl_util.envs.wrappers import VmapWrapper
from jax_rl_util.util.logging_util import DummyLogger, LoggableConfig, log_norms, with_logger

warnings.simplefilter(action="ignore", category=FutureWarning)

DISABLE_JIT = False
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"


@dataclass
class PPOParams(LoggableConfig):
    """Parameters for PPO."""

    # General settings
    project_name: str | None = "PPO RNN"
    logging: str = "aim"
    debug: int = 0
    seed: int = -1
    MODEL: str = "GRU"
    NUM_UNITS: int = 64
    meta_rl: bool = True
    act_dist_name: str = "normal"
    log_norms: bool = False
    record_best_eval_episode: bool = False

    # Training Settings
    episodes: int = 100000
    patience: int | None = 100
    eval_every: int = 10
    eval_steps: int = 1000
    eval_batch_size: int = 100
    collect_horizon: int = 20
    rollout_horizon: int = 10
    train_batch_size: int = 256
    update_steps: int = 32
    updates_per_batch: int = 4

    # Optimization settings
    LR: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gradient_clip: float | None = 1.0
    anneal_lr: bool = False

    # Env settings
    env_params: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            env_name="Acrobot-v1",
            batch_size=512,
        )
    )
    dt: float = 1.0
    normalize_obs: bool = True
    normalize_gae: bool = True


class LSTM(nn.Module):
    """Simple LSTM module."""

    config: Dict

    @functools.partial(
        nn.transforms.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False}
    )
    @nn.compact
    def __call__(self, carry, x):  # noqa
        features = carry[0].shape[-1]
        rnn_state = carry
        ins, resets = x
        rnn_state = jax.tree.map(
            lambda new, old: jnp.where(resets[:, None], new, old),
            self.initialize_carry(self.make_rng(), ins.shape),
            rnn_state,
        )
        return nn.OptimizedLSTMCell(features)(rnn_state, ins)

    def initialize_carry(self, rng, input_shape):
        """See flax dokumantation for more info."""
        return nn.OptimizedLSTMCell(self.config.NUM_UNITS, parent=None).initialize_carry(rng, input_shape)


class CTRNN(nn.Module):
    """Simple LSTM module."""

    config: PPOParams

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):  # noqa
        h = carry
        ins, resets = x
        h = jnp.where(
            resets[:, None],
            self.initialize_carry(None, ins.shape),
            h,
        )
        dense = nn.Dense(self.config.NUM_UNITS)
        tau = self.param("tau", functools.partial(jrandom.uniform, minval=1, maxval=10), (self.config.NUM_UNITS,))

        y = jnp.concatenate([ins, h], axis=-1)
        u = dense(y)
        act = jnp.tanh(u)
        dh = (act - h) / (1 + jax.nn.softplus(tau))
        out = jax.tree.map(lambda a, b: a + b * self.config.dt, h, dh)
        return out, out

    def initialize_carry(self, rng, input_shape):
        """See flax dokumantation for more info."""
        return jnp.zeros((*input_shape[:-1], self.config.NUM_UNITS))


class GRU(nn.Module):
    """GRU module."""

    config: PPOParams

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Apply the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(None, ins.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(carry[0].shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    def initialize_carry(self, rng, input_shape):
        """Use a dummy key since the default state init fn is just zeros."""
        return nn.GRUCell(
            self.config.NUM_UNITS, parent=None, param_dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        ).initialize_carry(rng, input_shape)


class LRU(nn.Module):
    """LRU module."""

    config: PPOParams
    d_hidden: int = 64

    @nn.compact
    def __call__(self, carry, x):
        """Apply the module."""
        from jax_rtrl.models.lru import OnlineLRULayer

        x = jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1), x)
        ins, resets = x
        carry, out = jax.vmap(OnlineLRULayer(self.config.NUM_UNITS, self.d_hidden))(carry, ins, resets)
        return carry, jnp.swapaxes(out, 0, 1)

    def initialize_carry(self, rng, input_shape):
        """Initialize the lru hidden state as zeros."""
        batch_size = input_shape[0:1] if len(input_shape) > 1 else ()
        hidden_init = jnp.zeros((*batch_size, self.d_hidden), dtype=jnp.complex64)
        return hidden_init


class MLP(nn.Module):
    """GRU module."""

    config: PPOParams

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Apply the module."""
        ins, resets = x
        y = nn.Dense(self.config.NUM_UNITS)(ins)
        y = nn.relu(y)
        y = nn.Dense(self.config.NUM_UNITS)(y)
        y = nn.relu(y)
        y = nn.Dense(self.config.NUM_UNITS)(y)
        return carry, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Use a dummy key since the default state init fn is just zeros."""
        return None


class ActorCriticRNN(nn.Module):
    """Actor Critic RNN."""

    action_dim: Sequence[int]
    discrete: bool
    config: PPOParams
    action_limits: jnp.ndarray = None

    def dist(self, model_out, log_std_min=0.01, log_std_max=None):
        """Split the output of the actor into mean and std.

        Applies squashing to the normal distribution

        Args:
        ----
            model_out : (jnp.ndarray):
                    Output of the actor
            log_std_min : (float):
                    Minimum value of the log std
            log_std_max : (float), optional:
                    Deprecated, unused! Maximum value of the log std

        Returns:
        -------
            jnp.ndarray: Mean of the distribution
            jnp.ndarray: Std of the distribution
        """
        if self.action_limits:
            # If action limits are defined we sample from [0, 1] and transform the event.
            act_range = jnp.array(self.action_limits[1]) - jnp.array(self.action_limits[0])
            act_min = jnp.array(self.action_limits[0])
            scaling_transform = tfp.bijectors.Chain([tfp.bijectors.Shift(act_min), tfp.bijectors.Scale(act_range)])

        if self.discrete:
            return tfp.distributions.Categorical(logits=model_out)
        else:
            if self.config.act_dist_name == "beta":
                alpha = jax.nn.softplus(model_out[..., : model_out.shape[-1] // 2])
                beta = jax.nn.softplus(model_out[..., model_out.shape[-1] // 2 :])
                return tfp.distributions.TransformedDistribution(tfp.distributions.Beta(alpha, beta), scaling_transform)
            elif self.config.act_dist_name == "brax":
                from brax.training.distribution import NormalTanhDistribution

                return NormalTanhDistribution(event_size=self.action_dim).create_dist(model_out)
            else:
                mean = model_out[..., : model_out.shape[-1] // 2]
                std = model_out[..., model_out.shape[-1] // 2 :]

                # if log_std_max is not None:
                #     # Squashed Gaussian taken from SAC
                #     # https://spinningup.openai.com/en/latest/algorithms/sac.html#id1
                #     std = jnp.tanh(std)
                #     std = log_std_min + 0.5 * (log_std_max - log_std_min) * (std + 1)
                # elif log_std_min is not None:
                #     std = jnp.clip(std, min=log_std_min)
                dist = tfp.distributions.Normal(mean, jax.nn.softplus(std) + log_std_min)
                if not self.action_limits:
                    return dist
                else:
                    tranforms = [scaling_transform, tfp.bijectors.Sigmoid()]
                    return tfp.distributions.TransformedDistribution(dist, tfp.bijectors.Chain(tranforms))

    @nn.compact
    def __call__(self, hidden, x):
        """Compute embedding from GRU and then actor and critic MLPs."""
        obs, dones = x
        embedding = nn.Dense(self.config.NUM_UNITS, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = globals()[self.config.MODEL](self.config)(hidden, rnn_in)

        actor_mean = nn.Dense(self.config.NUM_UNITS, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        action_dim = self.action_dim if self.discrete else self.action_dim * 2
        actor_mean = nn.Dense(action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        pi = self.dist(actor_mean)

        critic = nn.Dense(self.config.NUM_UNITS, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    def initialize_carry(self, rng, input_shape):
        """Initialize the rnn hidden state."""
        return globals()[self.config.MODEL](self.config).initialize_carry(rng, input_shape)


class Transition(NamedTuple):
    """A transition used in batch updates."""

    done: jnp.ndarray
    next_done: jnp.ndarray
    action: jnp.ndarray
    prev_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    prev_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    # next_obs: jnp.ndarray
    hidden: jnp.ndarray
    info: jnp.ndarray
    state: jnp.ndarray


def make_train(config: PPOParams, logger: DummyLogger):
    """Create the training function."""
    _rnn_model = globals()[config.MODEL](config)

    env, env_info, eval_env = make_env(config.env_params, make_eval=True)
    eval_env = VmapWrapper(eval_env, config.eval_batch_size)
    _discrete = env_info["discrete"]
    if env_info["act_clip"]:
        action_clip = jnp.array(env_info["act_clip"])
        # action_clip = jnp.nextafter(action_clip, action_clip.mean(axis=0))
        action_clip = (1 - 1e-4) * action_clip

    print_env_info(env_info)

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_size, discrete=_discrete, config=config, action_limits=env_info["act_clip"])
        rng, _rng = jax.random.split(rng)
        input_size = env.observation_size
        if config.meta_rl:
            # Previous action and reward are also inputs in MetaRL
            input_size += env.action_size + 1
        init_x = (
            jnp.zeros((1, config.env_params.batch_size, input_size)),
            jnp.zeros((1, config.env_params.batch_size)),
        )
        init_hstate = _rnn_model.initialize_carry(rng, (config.env_params.batch_size, input_size))

        network_params = network.init(_rng, init_hstate, init_x)

        if config.anneal_lr:
            linear_schedule = optax.linear_schedule(config.LR, 0.0, config.episodes * config.update_steps)
            tx = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip) if config.gradient_clip else optax.identity(),
                optax.adam(learning_rate=linear_schedule),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip) if config.gradient_clip else optax.identity(),
                # optax.sgd(config.LR),
                optax.adam(config.LR),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, reset_rng = jax.random.split(rng)
        env_state = env.reset(reset_rng)
        obsv = env_state.obs
        init_hstate = _rnn_model.initialize_carry(rng, obsv.shape)

        # Set up running statistics
        if config.normalize_obs:
            normalizer_state = running_statistics.init_state(obsv[0])
            normalize = running_statistics.normalize
        else:
            normalizer_state = None

            def normalize(x, y):
                return x  # noqa

        # Make Buffer
        buffer = fbx.make_trajectory_buffer(
            add_batch_size=config.env_params.batch_size,
            sample_batch_size=config.train_batch_size,
            sample_sequence_length=config.rollout_horizon,
            period=1,
            min_length_time_axis=config.collect_horizon,
            max_length_time_axis=config.collect_horizon,
        )

        @jax.jit
        def eval_model(params, _normalizer_state, seed=0):
            """Evaluate model."""
            print("Tracing eval_model.")
            rng = jax.random.PRNGKey(seed)
            rng, rng_init = jax.random.split(rng)

            env_state = eval_env.reset(rng_init)
            runner_state = (
                env_state,
                jnp.zeros((eval_env.batch_size, env.action_size)),
                _rnn_model.initialize_carry(rng_init, env_state.obs.shape),
                _rng,
            )
            # COLLECT TRAJECTORIES

            def _env_step(runner_state, unused):
                _env_state, last_act, prev_hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                x = normalize(_env_state.obs[None, :], _normalizer_state)
                if config.meta_rl:
                    x = jnp.concatenate(
                        [x, last_act[None], _env_state.reward.reshape((1, eval_env.batch_size, -1))],
                        axis=-1,
                    )
                ac_in = (x, _env_state.done[None, :])
                next_hstate, pi, value = network.apply(params, prev_hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                next_env_state = eval_env.step(_env_state, action)

                transition = Transition(
                    done=_env_state.done,
                    next_done=next_env_state.done,
                    action=action,
                    prev_action=last_act,
                    value=value,
                    reward=next_env_state.reward,
                    prev_reward=_env_state.reward,
                    log_prob=log_prob.mean(axis=-1) if not _discrete else log_prob,
                    obs=_env_state.obs,
                    # next_obs= # next_env_state.obs,
                    hidden=prev_hstate,
                    info=_env_state.info,
                    state=_env_state.pipeline_state,
                )

                # Action fed to the Meta-Learner is one-hot encoded for discrete envs.
                re_action = jax.nn.one_hot(action, env.action_size) if _discrete else action

                runner_state = (
                    next_env_state,
                    re_action,
                    next_hstate,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.eval_steps)

            # For episodes that are done early, get the first occurence of done
            ep_until = jnp.where(traj_batch.done.any(axis=0), traj_batch.done.argmax(axis=0), traj_batch.done.shape[0])
            # Compute cumsum and get value corresponding to end of episode per batch.
            ep_rewards = traj_batch.reward.cumsum(axis=0)[ep_until, jnp.arange(ep_until.shape[-1])]
            return ep_rewards.mean(), traj_batch

        # TRAIN LOOP
        @jax.jit
        def update_step(runner_state, unused):
            print("Tracing update_step.")

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, _normalizer_state, last_act, prev_hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                x = normalize(env_state.obs, _normalizer_state)
                if config.meta_rl:
                    x = jnp.concatenate([x, last_act, env_state.reward.reshape((env.batch_size, 1))], axis=-1)
                ac_in = (x[None], env_state.done[None, :])
                next_hstate, pi, value = network.apply(train_state.params, prev_hstate, ac_in)
                action = pi.sample(seed=_rng)
                if env_info["act_clip"]:
                    action = jnp.clip(action, *action_clip)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                next_env_state = env.step(env_state, action)
                next_env_state = next_env_state.replace(obs=normalize(next_env_state.obs, _normalizer_state))

                transition = Transition(
                    done=env_state.done,
                    next_done=next_env_state.done,
                    action=action,
                    prev_action=last_act,
                    value=value,
                    reward=next_env_state.reward,
                    prev_reward=env_state.reward,
                    log_prob=log_prob.mean(axis=-1) if not _discrete else log_prob,
                    obs=env_state.obs,
                    # next_obs= # next_env_state.obs,
                    hidden=prev_hstate,
                    info=env_state.info,
                    state=env_state.pipeline_state,
                )

                # Action fed to the Meta-Learner is one-hot encoded for discrete envs.
                re_action = jax.nn.one_hot(action, env.action_size) if _discrete else action

                runner_state = (
                    train_state,
                    next_env_state,
                    _normalizer_state,
                    re_action,
                    next_hstate,
                    rng,
                )
                return runner_state, transition

            # initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.collect_horizon)
            train_state, env_state, _normalizer_state, re_action, hstate, rng = runner_state

            # UPDATE NORMALIZER
            if config.normalize_obs:
                _normalizer_state = running_statistics.update(_normalizer_state, traj_batch.obs)

            # CALCULATE ADVANTAGES
            # Compute the last value
            x = normalize(env_state.obs, _normalizer_state)
            if config.meta_rl:
                x = jnp.concatenate([x, re_action, env_state.reward], axis=-1)
            ac_in = (x[None], env_state.done[None, :])
            _, _, _last_val = network.apply(train_state.params, hstate, ac_in)

            def _calculate_gae(transitions, values, last_val):
                """Compute the generalized advantage estimates."""

                def _get_advantages(carry, _batch: tuple[Transition, jax.Array]):
                    gae, next_value = carry
                    _transition, _value = _batch
                    next_done, reward = _transition.next_done, _transition.reward.squeeze()
                    delta = reward + config.gamma * next_value * (1 - next_done) - _value
                    gae = delta + config.gamma * config.gae_lambda * (1 - next_done) * gae
                    return (gae, _value), gae

                _, advantages = jax.lax.scan(
                    jax.vmap(_get_advantages),
                    (jnp.zeros_like(last_val), last_val),
                    (transitions, values),
                    reverse=True,
                    unroll=config.rollout_horizon,
                )
                return advantages, advantages + transitions.value

            gae, val = _calculate_gae(traj_batch, traj_batch.value, _last_val[0])

            # Swap axes to make batch major
            batch_major = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), (traj_batch, gae, val))
            # Add to buffer
            buffer_state = buffer.init(jax.tree.map(lambda x: x[0][0], batch_major))
            buffer_state = buffer.add(buffer_state, batch_major)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (train_state, rng) = update_state

                def _update_minbatch(train_state: TrainState, batch: tuple[Transition, jax.Array, jax.Array]):
                    def _loss_fn(params):
                        transition, _gae, _val = batch
                        _init_hstate = jax.tree.map(lambda a: a[0], transition.hidden)  # T=0, B, H
                        # RERUN NETWORK
                        x = normalize(transition.obs, _normalizer_state)
                        if config.meta_rl:
                            x = jnp.concatenate(
                                [x, transition.prev_action, transition.prev_reward.reshape((*x.shape[:2], 1))],
                                axis=-1,
                            )
                        _, pi, _values = network.apply(params, _init_hstate, (x, transition.done))

                        if env_info["act_clip"]:
                            action = jnp.clip(transition.action, *action_clip)
                        else:
                            action = transition.action
                        log_prob = pi.log_prob(action)
                        if not _discrete:
                            log_prob = log_prob.mean(axis=-1)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = transition.value + (_values - transition.value).clip(
                            -config.clip_eps, config.clip_eps
                        )
                        value_losses = jnp.square(_val - _values)
                        value_losses_clipped = jnp.square(_val - value_pred_clipped)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        # value_loss = 0.5 * value_losses.mean()

                        # CALCULATE ACTOR LOSS
                        # Cumsum of log_probs since they depend on previous actions
                        diff = log_prob.cumsum(axis=0) - transition.log_prob.cumsum(axis=0)
                        diff = jnp.clip(diff, max=10)  # HACK avoids some NaNs!
                        ratio = jnp.exp(diff)
                        if config.normalize_gae:
                            _gae = (_gae - _gae.mean()) / (_gae.std() + 1e-8)
                        loss_actor1 = ratio * _gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * _gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        total_loss = loss_actor + config.vf_coef * value_loss

                        if not _discrete and config.act_dist_name == "normal":
                            entropy = pi.distribution.entropy().mean()
                        else:
                            entropy = pi.entropy().mean()
                        if config.ent_coef:
                            total_loss -= config.ent_coef * entropy
                        return total_loss, {"value_loss": value_loss, "loss_actor": loss_actor, "entropy": entropy}

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, loss_info), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng)
                experience = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatch.experience)

                # batch_indices = jrandom.choice(
                #     _rng, config.env_params.batch_size, (config.train_batch_size,), replace=False
                # )
                # experience = jax.tree.map(lambda x: x[batch_indices], batch_major)

                # Swap axes back to time major
                # experience = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), experience)

                train_state, loss_info = _update_minbatch(train_state, experience)
                update_state = (train_state, rng)
                return update_state, loss_info

            update_state = (train_state, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.updates_per_batch)
            runner_state = (update_state[0], env_state, _normalizer_state, re_action, hstate, update_state[-1])
            return runner_state, loss_info

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            normalizer_state,
            jnp.zeros((config.env_params.batch_size, env.action_size)),
            init_hstate,
            _rng,
        )

        steps_since_best = 0
        trajectories: Transition = None
        try:
            for i in range(config.episodes):
                with jax.disable_jit(DISABLE_JIT):
                    runner_state, loggables = jax.lax.scan(update_step, runner_state, None, config.update_steps)
                if i % config.eval_every == 0:
                    eval_reward, _traj = eval_model(runner_state[0].params, runner_state[2])

                    timestep = runner_state[0].step  # * config.collect_horizon * config.env_params.batch_size
                    loggables = {
                        **jax.tree.map(jnp.mean, loggables),
                        "eval/rewards": eval_reward,
                        "runner_step": timestep,
                    }
                    if config.log_norms:
                        loggables.update(**log_norms(runner_state[0].params)[0])
                    logger.flush()
                    if eval_reward > float(logger["best_eval_reward"]):
                        steps_since_best = 0
                        logger["best_eval_reward"] = float(eval_reward)
                        loggables["best_eval_reward"] = eval_reward
                        trajectories = _traj
                    else:
                        steps_since_best += 1
                    logger.log(loggables)
                    print(
                        f"Global step: {timestep:2.0e}, eval reward: {eval_reward:.2f}, best: {logger['best_eval_reward']:.2f}"
                    )
                    # Early stopping
                    if config.patience and steps_since_best >= config.patience:
                        print(f"Early stopping patience {config.patience}")
                        break
        except KeyboardInterrupt:
            print("Interrupted by user, Finalizing...")
        finally:
            if config.record_best_eval_episode and trajectories is not None:
                # Save last episode data for plotting.
                # Swap axes to batch major
                trajectories = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectories)

                out_dir = f"data/{config.env_params.env_name}"
                os.makedirs(out_dir, exist_ok=True)
                data = {
                    "obs": trajectories.obs,
                    "action": trajectories.action,
                    "reward": trajectories.reward,
                    "done": trajectories.next_done,
                    **trajectories.info,
                    **trajectories.state,
                }
                np.savez(f"{out_dir}/ppo_best_trajectory.npz", **data)
                env_params = getattr(env, "params")
                if env_params:
                    with open(f"{out_dir}/ppo_env_params.pkl", "wb") as f:
                        pickle.dump(env_params, f)
        return logger["best_eval_reward"]

    return train


def train_and_eval(config: PPOParams, logger=DummyLogger()):
    """Run training."""
    rng = jax.random.PRNGKey(config.seed)
    logger["best_eval_reward"] = -np.inf
    try:
        result = make_train(config, logger)(rng)

        if config.env_params.env_name == "dronegym":
            # CUSTOM Plotting
            out_dir = f"data/{config.env_params.env_name}"
            plot_from_file(f"{out_dir}/ppo_best_trajectory.npz", f"{out_dir}/ppo_env_params.pkl", "best")
            logger.log_img("best_trajectory", plt.gcf())
        return result
    except Exception as e:
        raise e
    finally:
        logger.finalize()


if __name__ == "__main__":
    config_path = "config/ppo.yaml" if os.path.exists("config/ppo.yaml") else None
    params: PPOParams = simple_parsing.parse(PPOParams, config_path=config_path)
    best_reward = with_logger(train_and_eval, params, run_name=params.env_params.env_name)
    print(f"Best eval reward: {best_reward:.2f}")
