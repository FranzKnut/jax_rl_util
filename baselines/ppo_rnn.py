"""PPO implementation in JAX."""

import functools
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Sequence

import distrax
import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import simple_parsing
from brax.training.acme import running_statistics
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from jax_rl_util.envs.environments import EnvironmentConfig, make_env, print_env_info
from jax_rl_util.envs.wrappers import VmapWrapper
from jax_rl_util.util.logging_util import DummyLogger, LoggableConfig, log_norms, with_logger
from jax_rtrl.models.lru import OnlineLRULayer

warnings.simplefilter(action="ignore", category=FutureWarning)

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"


@dataclass
class PPOParams(LoggableConfig):
    """Parameters for PPO."""

    # General settings
    project_name: str | None = "PPO RNN"
    logging: str = "aim"
    debug: int = 0
    seed: int = 0
    MODEL: str = "LRU"
    NUM_UNITS: int = 64
    meta_rl: bool = False
    act_dist_name: str = "normal"
    log_norms: bool = False

    # Training Settings
    episodes: int = 100000
    patience: int = 100
    eval_every: int = 1
    eval_steps: int = 1000
    eval_batch_size: int = 10
    collect_horizon: int = 20
    rollout_horizon: int = 10
    train_batch_size: int = 32
    update_steps: int = 10
    UPDATE_EPOCHS: int = 4

    # Optimization settings
    LR: float = 3e-4
    gamma: float = 0.99
    GAE_LAMBDA: float = 0.9
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.001
    VF_COEF: float = 0.5
    gradient_clip: float | None = 1.0
    ANNEAL_LR: bool = False

    # Env settings
    env_params: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig(batch_size=64))
    dt: float = 1.0
    normalize: bool = True
    eps: float = 0


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
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        return nn.OptimizedLSTMCell(features)(rnn_state, ins)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        """See flax dokumantation for more info."""
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_dims, hidden_size)
        )


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
        dh = (act - h) / tau
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
        return nn.GRUCell(self.config.NUM_UNITS, parent=None).initialize_carry(rng, input_shape)


class LRU(nn.Module):
    """LRU module."""

    config: PPOParams
    d_hidden: int = 64

    @nn.compact
    def __call__(self, carry, x):
        """Apply the module."""
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

    def dist(self, model_out, log_std_min=-5, log_std_max=None):
        """Split the output of the actor into mean and std.

        Applies squashing to the normal distribution

        Args:
        ----
            model_out : (jnp.ndarray):
                    Output of the actor
            log_std_min : (float):
                    Minimum value of the log std
            log_std_max : (float), optional:
                    Maximum value of the log std

        Returns:
        -------
            jnp.ndarray: Mean of the distribution
            jnp.ndarray: Std of the distribution
        """
        if self.discrete:
            return distrax.Categorical(logits=model_out)
        else:
            if self.config.act_dist_name == "beta":
                alpha = jax.nn.softplus(model_out[..., : model_out.shape[-1] // 2])
                beta = jax.nn.softplus(model_out[..., model_out.shape[-1] // 2 :])
                act_range = jnp.array(self.action_limits[1]) - jnp.array(self.action_limits[0])
                return distrax.Transformed(
                    distrax.Beta(alpha, beta), distrax.ScalarAffine(jnp.array(self.action_limits[0]), act_range)
                )
            elif self.config.act_dist_name == "brax":
                from brax.training.distribution import NormalTanhDistribution

                return NormalTanhDistribution(event_size=self.action_dim).create_dist(model_out)
            else:
                mean = model_out[..., : model_out.shape[-1] // 2]
                log_std = model_out[..., model_out.shape[-1] // 2 :]

                if log_std_max is not None:
                    # Squashed Gaussian taken from SAC
                    # https://spinningup.openai.com/en/latest/algorithms/sac.html#id1
                    log_std = jnp.tanh(log_std)
                    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
                else:
                    log_std = jnp.clip(log_std, a_min=log_std_min)
                return distrax.Normal(mean, jnp.exp(log_std))

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
    next_obs: jnp.ndarray
    hidden: jnp.ndarray
    info: jnp.ndarray


def make_train(config: PPOParams, logger: DummyLogger):
    """Create the training function."""
    _rnn_model = globals()[config.MODEL](config)

    env, env_info, eval_env = make_env(config.env_params, make_eval=True)
    eval_env = VmapWrapper(eval_env, config.eval_batch_size)
    _discrete = env_info["discrete"]

    print_env_info(env_info)

    def train(rng, record_best_eval_episode=False):
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

        if config.ANNEAL_LR:
            linear_schedule = optax.linear_schedule(1.0, 0.0, config.episodes * config.update_steps)
            tx = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip) if config.gradient_clip else optax.identity(),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
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
        if config.normalize:
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
                log_prob = pi.log_prob(action * (1 - config.eps))
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                next_env_state = eval_env.step(_env_state, action)
                _reward = next_env_state.reward.reshape((eval_env.batch_size, 1))

                next_env_state = next_env_state.replace(obs=normalize(next_env_state.obs, _normalizer_state))

                transition = Transition(
                    _env_state.done,
                    next_env_state.done,
                    action,
                    last_act,
                    value,
                    _reward,
                    _env_state.reward,
                    log_prob,
                    _env_state.obs,
                    next_env_state.obs,
                    prev_hstate,
                    _env_state.info,
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
            ep_until = jnp.where(traj_batch.done.any(axis=0), traj_batch.done.argmax(axis=0), -1)
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
                log_prob = pi.log_prob(action * (1 - config.eps))
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                next_env_state = env.step(env_state, action)
                _reward = next_env_state.reward.reshape((env.batch_size, 1))

                next_env_state = next_env_state.replace(obs=normalize(next_env_state.obs, _normalizer_state))

                transition = Transition(
                    env_state.done,
                    next_env_state.done,
                    action,
                    last_act,
                    value,
                    _reward,
                    env_state.reward,
                    log_prob.mean(axis=-1) if not _discrete else log_prob,
                    env_state.obs,
                    next_env_state.obs,
                    prev_hstate,
                    env_state.info,
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
            train_state, env_state, _normalizer_state, last_act, hstate, rng = runner_state

            # UPDATE NORMALIZER
            if config.normalize:
                _normalizer_state = running_statistics.update(_normalizer_state, traj_batch.obs)

            # Swap axes to make batch major
            batch_major = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
            # Add to buffer
            buffer_state = buffer.init(jax.tree.map(lambda x: x[0][0], batch_major))
            buffer_state = buffer.add(buffer_state, batch_major)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (train_state, rng) = update_state

                def _update_minbatch(train_state, _batch: Transition):
                    def _loss_fn(params):
                        # CALCULATE ADVANTAGES
                        # Compute the last value
                        x = normalize(_batch.next_obs[-1:], _normalizer_state)
                        last_hidden = jax.tree.map(lambda a: a[-1], _batch.hidden)
                        re_action = (
                            jax.nn.one_hot(_batch.action[-1:], env.action_size) if _discrete else _batch.action[-1:]
                        )

                        if config.meta_rl:
                            x = jnp.concatenate([x, re_action, _batch.reward[-1:]], axis=-1)
                        ac_in = (x, _batch.done[-1:])
                        _, _, _last_val = network.apply(train_state.params, last_hidden, ac_in)

                        def _calculate_gae(_batch, values, last_val, last_done):
                            """Compute the generalized advantage estimates."""

                            def _get_advantages(carry, _batch):
                                gae, next_value, next_done = carry
                                transition, _value = _batch
                                done, value, reward = transition.done, _value, transition.reward.squeeze()
                                delta = reward + config.gamma * next_value * (1 - next_done) - value
                                gae = delta + config.gamma * config.GAE_LAMBDA * (1 - next_done) * gae
                                return (gae, value, done), gae

                            _, advantages = jax.lax.scan(
                                jax.vmap(_get_advantages),
                                (jnp.zeros_like(last_val), last_val, last_done),
                                (_batch, values),
                                reverse=True,
                                unroll=config.rollout_horizon,
                            )
                            return advantages, advantages + _batch.value

                        _init_hstate = jax.tree.map(lambda a: a[0], _batch.hidden)  # T=0, B, H
                        # RERUN NETWORK
                        x = normalize(_batch.obs, _normalizer_state)
                        if config.meta_rl:
                            x = jnp.concatenate(
                                [x, _batch.prev_action, _batch.prev_reward.reshape((*x.shape[:2], 1))],
                                axis=-1,
                            )
                        _, pi, _value = network.apply(params, _init_hstate, (x, _batch.done))
                        log_prob = pi.log_prob(_batch.action * (1 - config.eps))
                        if not _discrete:
                            log_prob = log_prob.mean(axis=-1)

                        gae, target_value = _calculate_gae(_batch, _value, _last_val.squeeze(0), _batch.next_done[-1])

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = _batch.value + (_value - _batch.value).clip(
                            -config.CLIP_EPS, config.CLIP_EPS
                        )
                        value_losses = jnp.square(target_value - _value)
                        value_losses_clipped = jnp.square(value_pred_clipped - target_value)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - _batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy
                        return total_loss, {"value_loss": value_loss, "loss_actor": loss_actor, "entropy": entropy}

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, loss_info), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                rng, _rng = jax.random.split(rng)
                minibatches = buffer.sample(buffer_state, _rng)

                # Swap axes back to time major
                experience = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatches.experience)

                train_state, loss_info = _update_minbatch(train_state, experience)
                update_state = (train_state, rng)
                return update_state, loss_info

            update_state = (train_state, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.UPDATE_EPOCHS)
            runner_state = (update_state[0], env_state, _normalizer_state, last_act, hstate, update_state[-1])
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
        trajectories = None
        try:
            for i in range(config.episodes):
                runner_state, loggables = jax.lax.scan(update_step, runner_state, None, config.update_steps)
                if i % config.eval_every == 0:
                    eval_reward, _traj = eval_model(runner_state[0].params, runner_state[2])

                    timestep = runner_state[0].step * config.collect_horizon * config.env_params.batch_size
                    loggables = {
                        **jax.tree.map(jnp.mean, loggables),
                        "eval/rewards": eval_reward,
                        "global_steps": timestep,
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
        if record_best_eval_episode:
            # Save last episode data for plotting.
            os.makedirs("output", exist_ok=True)
            data = {
                "state": trajectories.info["state"],
                "time": np.arange(trajectories[0].shape[0]),
                "action": trajectories.action,
            }
            np.savez("output/trajectories.npz", **data)
        return logger["best_eval_reward"]

    return train


def train_and_eval(config: PPOParams, logger=DummyLogger()):
    """Run training."""
    rng = jax.random.PRNGKey(config.seed)
    logger["best_eval_reward"] = -np.inf
    try:
        return make_train(config, logger)(rng)
    except Exception as e:
        raise e
    finally:
        logger.finalize()


if __name__ == "__main__":
    params: PPOParams = simple_parsing.parse(PPOParams)
    best_reward = with_logger(train_and_eval, params, run_name=params.env_params.env_name)
    print(f"Best eval reward: {best_reward:.2f}")
