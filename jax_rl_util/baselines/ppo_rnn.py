"""PPO implementation in JAX."""

import copy
import functools
import os
from dataclasses import dataclass, field
import pickle
from typing import NamedTuple, Sequence

import brax
import distrax
import flax.linen as nn
import flashbax as fbx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_rtrl.util.checkpointing import (
    checkpointing,
    restore_config,
    restore_remote,
)
from matplotlib import pyplot as plt
import numpy as np
import optax
import simple_parsing
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tqdm import trange

from jax_rl_util.util.logging_util import (
    DummyLogger,
    LoggableConfig,
    log_norms,
    update_nested_dict,
    with_logger,
)

from jax_rl_util.envs.env_util import compute_agg_reward, render_frames
from jax_rl_util.envs.environments import (
    EnvironmentConfig,
    make_wrapped_env,
    print_env_info,
)
from jax_rl_util.envs.wrappers import VmapWrapper
from jax_rl_util.optimizers import OptimizerConfig, make_optimizer_for_model
from jax_rl_util.util import running_statistics

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true"

# Enable Caching
jax.config.update("jax_compilation_cache_dir", "tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", 10)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


@dataclass
class PPOParams(LoggableConfig):
    """Parameters for PPO."""

    project_name: str | None = "PPO-RNN"
    logging: str | None = None
    debug: int = 0
    seed: int = -1
    log_norms: bool = False
    save_model: bool = False
    ckpt_path: str | None = None  # path or wandb run id
    fresh: bool = True  # Only load the hyperparameters, not the model
    record_best_eval_episode: bool = True
    deterministic_eval: bool = True

    # Model Settings
    model: str = "CTRNN"
    dt: float = 1.0
    num_units: int = 256
    meta_rl: bool = False
    act_dist_name: str = "normal"
    log_norms: bool = False
    record_best_eval_episode: bool = False

    # Training Settings
    episodes: int = 1000
    patience: int = 400
    eval_every: int = 1
    render_every_evals: int = 1
    render: bool = False
    render_start: int = 0
    render_steps: int = 200
    eval_steps: int = 5000
    eval_batch_size: int = 10
    collect_steps: int = 100
    rollout_horizon: int = 20
    train_batch_size: int = 256
    update_steps: int = 10
    update_epochs: int = 4
    fixed_env_rng: bool = False

    # Optimization settings
    optimizer_params: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            opt_name="adam",
            learning_rate=3e-4,
        )
    )
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 1e-5
    vf_coef: float = 0.5
    anneal_ent: bool = False

    # Env settings
    env_params: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            env_name="CartPole-v1",
            batch_size=128,
        )
    )
    dt: float = 1.0
    normalize_obs: bool = False
    normalize_gae: bool = False
    sparsity_penalty: float | None = None


def normalize_legacy_optimizer_config(config_dict: dict) -> dict:
    """Map legacy optimizer keys to optimizer_params."""
    config_dict = dict(config_dict)
    optimizer_params = config_dict.get("optimizer_params", {})
    if not isinstance(optimizer_params, dict):
        optimizer_params = dict(vars(optimizer_params))
    else:
        optimizer_params = dict(optimizer_params)

    legacy_lr = config_dict.pop("LR", None)
    legacy_gradient_clip = config_dict.pop("gradient_clip", None)
    legacy_anneal_lr = config_dict.pop("anneal_lr", None)

    if legacy_lr is not None and "learning_rate" not in optimizer_params:
        optimizer_params["learning_rate"] = legacy_lr
    if legacy_gradient_clip is not None and "gradient_clip" not in optimizer_params:
        optimizer_params["gradient_clip"] = legacy_gradient_clip
    if legacy_anneal_lr:
        print(
            "WARNING: anneal_lr is deprecated. Configure optimizer_params.lr_decay_type and optimizer_params.lr_kwargs instead."
        )

    config_dict["optimizer_params"] = optimizer_params
    return config_dict


class LSTM(nn.Module):
    """Simple LSTM module."""

    config: dict

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):  # noqa
        features = carry[0].shape[-1]
        rnn_state = carry
        ins, resets = x
        rnn_state = jax.tree.map(
            lambda new, old: jnp.where(resets[:, None], new, old),
            self.initialize_carry(jrandom.PRNGKey(0), ins.shape),
            rnn_state,
        )
        return nn.OptimizedLSTMCell(features)(rnn_state, ins)

    def initialize_carry(self, rng, input_shape):
        """See flax dokumantation for more info."""
        return nn.OptimizedLSTMCell(
            self.config.NUM_UNITS, parent=None
        ).initialize_carry(rng, input_shape)


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
        dense = nn.Dense(self.config.num_units)
        tau = self.param(
            "tau",
            functools.partial(jrandom.uniform, minval=2, maxval=8),
            (self.config.num_units,),
        )

        y = jnp.concatenate([ins, h], axis=-1)
        u = dense(y)
        act = jnp.tanh(u)
        dh = (act - h) / (1 + jax.nn.softplus(tau))
        out = jax.tree.map(lambda a, b: a + b * self.config.dt, h, dh)
        return out, out

    def initialize_carry(self, rng, input_shape):
        """See flax dokumantation for more info."""
        return jnp.zeros((*input_shape[:-1], self.config.num_units))


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
            self.config.num_units,
            parent=None,
            param_dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32,
        ).initialize_carry(rng, input_shape)


class LRU(nn.Module):
    """LRU module."""

    config: PPOParams
    d_hidden: int = 64
    num_layers: int = 1

    @nn.compact
    def __call__(self, carry, x):
        """Apply the module."""
        from jax_rtrl.models.cells.lru import OnlineLRULayer

        x = jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1), x)
        ins, resets = x
        model = OnlineLRULayer(self.config.num_units, self.d_hidden)
        carry, out = jax.vmap(model)(carry, ins, resets=resets)
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
        y = nn.Dense(self.config.num_units)(ins)
        y = nn.relu(y)
        y = nn.Dense(self.config.num_units)(y)
        # y = nn.relu(y)
        # y = nn.Dense(self.config.NUM_UNITS)(y)
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
    log_std_min: jnp.ndarray = 0.001
    log_std_max: jnp.ndarray = 2

    def dist(self, model_out):
        """Split the output of the actor into mean and std.

        Applies squashing to the normal distribution

        Args:
        ----
            model_out : (jnp.ndarray):
                    Output of the actor

        Returns:
        -------
            jnp.ndarray: Mean of the distribution
            jnp.ndarray: Std of the distribution
        """
        if self.action_limits:
            # If action limits are defined we sample from [0, 1] and transform the event.
            act_range = jnp.array(self.action_limits[1]) - jnp.array(
                self.action_limits[0]
            )
            act_min = jnp.array(self.action_limits[0])
            scaling_transform = distrax.ScalarAffine(act_min, act_range)

        if self.discrete:
            return distrax.Categorical(logits=model_out)
        else:
            if self.config.act_dist_name == "beta":
                alpha = jax.nn.softplus(model_out[..., : model_out.shape[-1] // 2])
                beta = jax.nn.softplus(model_out[..., model_out.shape[-1] // 2 :])
                return distrax.Transformed(distrax.Beta(alpha, beta), scaling_transform)
            else:
                if self.config.act_dist_name == "normal_scale":
                    mean = model_out[..., : model_out.shape[-1] // 2]
                    log_std = model_out[..., model_out.shape[-1] // 2 :]
                else:
                    mean = model_out
                    log_std = self.param(
                        "log_std", nn.initializers.constant(-1), (self.action_dim)
                    )
                #
                #     # Squashed Gaussian taken from SAC
                #     # https://spinningup.openai.com/en/latest/algorithms/sac.html#id1
                #     std = jnp.tanh(std)
                #     std = log_std_min + 0.5 * (log_std_max - log_std_min) * (std + 1)
                if self.log_std_min is not None:
                    log_std = jnp.clip(log_std, min=self.log_std_min)
                # log_std = jnp.array([-2] * self.action_dim)
                dist = distrax.LogStddevNormal(mean, log_std, self.log_std_max)
                return dist

    @nn.compact
    def __call__(self, hidden, x):
        """Compute embedding from RNN and then actor and critic MLPs."""
        obs, dones = x

        if hidden is None:
            hidden = self.initialize_carry(None, obs.shape[1:])

        action_dim = (
            self.action_dim * 2
            if not self.discrete and self.config.act_dist_name not in ["beta", "normal"]
            else self.action_dim
        )

        if self.config.model == "MLP_min":
            # embedding = jnp.tanh(obs)
            embedding = obs
            # no RNN
        else:
            embedding = nn.Dense(
                self.config.num_units,
                kernel_init=orthogonal(np.sqrt(np.sqrt(2))),
                bias_init=constant(0.0),
                name="emb",
            )(obs)
            embedding = nn.relu(embedding)

            rnn_in = (embedding, dones)
            hidden, embedding = globals()[self.config.model](self.config)(
                hidden, rnn_in
            )

            actor_mean = nn.Dense(
                self.config.num_units,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name="actor0",
            )(embedding)
            actor_mean = nn.relu(actor_mean)
        action_dim = (
            self.action_dim
            if self.discrete or self.config.act_dist_name == "normal"
            else self.action_dim * 2
        )
        actor_mean = nn.Dense(
            action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor1",
        )(embedding)

        # Compute the action distribution
        actor_mean = jnp.tanh(actor_mean)
        pi = self.dist(actor_mean)

        critic = nn.Dense(
            self.config.num_units,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic0",
        )(embedding)
        critic = nn.Dense(
            self.config.num_units,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic1",
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic2"
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @nn.nowrap
    def initialize_carry(self, rng, input_shape):
        """Initialize the rnn hidden state."""
        return globals()[self.config.model](self.config).initialize_carry(
            rng, input_shape
        )


class Transition(NamedTuple):
    """A transition used in batch updates."""

    prev_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    prev_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    prev_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    hidden: jnp.ndarray
    info: jnp.ndarray
    # state: jnp.ndarray
    env_state: brax.envs.State = None
    next_env_state: brax.envs.State = None


def calculate_gae(transitions, values, last_val, gamma=0.99, gae_lambda=0.95):
    """Compute the generalized advantage estimates."""

    def _get_advantages(carry, _batch: tuple[Transition, jax.Array]):
        gae, next_value = carry
        _transition, _value = _batch
        next_done, reward = _transition.done, _transition.reward.squeeze()
        delta = reward + gamma * next_value * (1 - next_done) - _value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, _value), gae

    _, advantages = jax.lax.scan(
        jax.vmap(_get_advantages),
        (jnp.zeros_like(last_val), last_val),
        (transitions, values),
        reverse=True,
        # unroll=rollout_horizon,
    )
    return advantages, advantages + transitions.value


def make_train(
    config: PPOParams,
    logger: DummyLogger,
    param_overrides=None,
    network_cls=None,
    obs_transform_fn=None,
    act_transform_fn=None,
):
    """Create the training function."""
    if network_cls is None:
        network_cls = ActorCriticRNN

    env, env_info, eval_env = make_wrapped_env(config.env_params, make_eval=True)
    eval_env = VmapWrapper(eval_env, config.eval_batch_size)
    _discrete = env_info["discrete"]
    if env_info["act_clip"]:
        action_clip = jnp.array(env_info["act_clip"])
        # action_clip = jnp.nextafter(action_clip, action_clip.mean(axis=0))
        if config.act_dist_name == "beta":
            action_clip = (1 - 1e-4) * action_clip

    print_env_info(env_info)

    def train(key_main):
        """Train Actor and Critic.

        Parameters
        ----------
        rng : PRNGKey
            jax random key

        Returns
        -------
            Average eval reward of last validation epoch
        """
        nonlocal config
        # INIT NETWORK
        network = network_cls(
            env.action_size,
            discrete=_discrete,
            config=config,
            action_limits=env_info["act_clip"],
        )

        _rng, init_rng, reset_rng = jax.random.split(key_main, 3)
        if config.fixed_env_rng:
            print(
                "WARNING: Using fixed environment RNG. This is not recommended for fresh training runs."
            )
            reset_rng = key_main
        tmp_env_state = env.reset(reset_rng)
        init_obs = jnp.zeros(
            (1,) + tmp_env_state.obs.shape,
            dtype=tmp_env_state.obs.dtype,
        )
        if config.meta_rl:
            # Previous action and reward are also inputs in MetaRL
            zeros_act = jnp.zeros(
                init_obs.shape[:-1] + (env.action_size,),
                dtype=init_obs.dtype,
            )
            zeros_rew = jnp.zeros(
                init_obs.shape[:-1] + (1,),
                dtype=init_obs.dtype,
            )
            init_obs = jnp.concatenate([init_obs, zeros_act, zeros_rew], axis=-1)
        init_x = (
            init_obs,
            jnp.zeros(
                (1,) + tmp_env_state.done.shape,
                dtype=tmp_env_state.done.dtype,
            ),
        )
        # Initialize once to have a tree template for robust restores.
        init_params = network.init(init_rng, None, init_x)

        # Set up checkpointing (match RTRRLAgent restore behavior).
        ckpt_path = config.ckpt_path or f"output/{config.model}-{config.num_units}"
        if ckpt_path.startswith("wandb:"):
            ckpt_path = restore_remote(ckpt_path)

        (network_params, restored_cfg), save_model = checkpointing(
            ckpt_path, config.fresh, config, tree=init_params
        )
        if restored_cfg:
            restored_cfg = normalize_legacy_optimizer_config(restored_cfg)
            if "ckpt_path" not in restored_cfg:
                restored_cfg["ckpt_path"] = config.ckpt_path
            config = PPOParams.from_dict(restored_cfg, drop_extra_fields=True)

        if network_params is None:
            network_params = init_params

        if param_overrides is not None:
            network_params = update_nested_dict(
                network_params, copy.deepcopy(param_overrides)
            )

        optimizer_config = config.optimizer_params
        if isinstance(optimizer_config, dict):
            optimizer_config = OptimizerConfig(**optimizer_config)
        tx = make_optimizer_for_model(config.model.lower(), optimizer_config)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        if config.fixed_env_rng:
            reset_rng = key_main
        else:
            _rng, reset_rng = jax.random.split(_rng)
        env_state = env.reset(reset_rng)
        obsv = env_state.obs
        init_hstate = network.apply(
            init_params, _rng, obsv.shape, method=network.initialize_carry
        )

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
            min_length_time_axis=config.rollout_horizon,
            max_length_time_axis=config.collect_steps,
        )

        @jax.jit
        def eval_model(
            params, _normalizer_state, seed=None
        ) -> tuple[jnp.ndarray, Transition]:
            """Evaluate model."""
            print("Tracing eval_model.")
            if seed is None:
                _rng = key_main
            else:
                _rng = jax.random.PRNGKey(seed)
            if config.fixed_env_rng:
                rng_init = _rng
            else:
                _rng, rng_init = jax.random.split(_rng)

            print("Evaluating with key:", rng_init)
            env_state = eval_env.reset(rng_init)
            # Normalize observations
            env_state = env_state.replace(
                obs=normalize(env_state.obs, _normalizer_state)
            )  # TODO: don't write normalized obs back to env_state, only use for policy input
            runner_state = (
                env_state,
                jnp.zeros((eval_env.batch_size, env.action_size)),
                network.apply(
                    init_params,
                    rng_init,
                    env_state.obs.shape,
                    method=network.initialize_carry,
                ),
                _rng,
            )
            # COLLECT TRAJECTORIES

            def _env_step(runner_state, unused):
                _env_state, last_act, prev_hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                x = _env_state.obs[None, :]
                if obs_transform_fn is not None:
                    x = obs_transform_fn(x)
                if config.meta_rl:
                    x = jnp.concatenate(
                        [
                            x,
                            last_act[None],
                            _env_state.reward.reshape((1, eval_env.batch_size, -1)),
                        ],
                        axis=-1,
                    )
                ac_in = (x, _env_state.done[None, :])
                next_hstate, pi, value = network.apply(
                    params, prev_hstate, ac_in, rngs={"default": _rng}
                )
                if config.deterministic_eval:
                    action = pi.mode()
                else:
                    action = pi.sample(seed=_rng)
                if act_transform_fn is not None:
                    action = act_transform_fn(action)
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
                next_env_state = eval_env.step(_env_state, action)

                transition = Transition(
                    env_state=_env_state,
                    next_env_state=next_env_state,
                    prev_done=_env_state.done,
                    done=next_env_state.done,
                    action=action,
                    prev_action=last_act,
                    value=value,
                    reward=next_env_state.reward,
                    prev_reward=_env_state.reward,
                    log_prob=log_prob,
                    obs=_env_state.obs,
                    # next_obs= # next_env_state.obs,
                    hidden=prev_hstate,
                    info=_env_state.info,
                    # state=_env_state.pipeline_state,
                )

                # Action fed to the Meta-Learner is one-hot encoded for discrete envs.
                re_action = (
                    jax.nn.one_hot(action, env.action_size) if _discrete else action
                )

                runner_state = (
                    next_env_state,
                    re_action,
                    next_hstate,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.eval_steps
            )
            mean_reward = compute_agg_reward(traj_batch)
            return mean_reward, traj_batch

        # TRAIN LOOP
        @jax.jit
        def update_step(runner_state, epoch):
            print("Tracing update_step.")

            if config.anneal_ent:
                entropy_schedule = optax.linear_schedule(
                    config.ent_coef, 0.0, config.episodes * config.update_steps
                )
            else:
                entropy_schedule = lambda _: config.ent_coef  # noqa

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    _normalizer_state,
                    last_act,
                    prev_hstate,
                    rng,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                x = normalize(env_state.obs, _normalizer_state)
                if obs_transform_fn is not None:
                    x = obs_transform_fn(x)
                if config.meta_rl:
                    x = jnp.concatenate(
                        [x, last_act, env_state.reward.reshape((env.batch_size, 1))],
                        axis=-1,
                    )
                ac_in = (x[None], env_state.done[None, :])
                next_hstate, pi, value = network.apply(
                    train_state.params, prev_hstate, ac_in, rngs={"default": _rng}
                )
                action = pi.sample(seed=_rng)
                if act_transform_fn is not None:
                    action = act_transform_fn(action)
                if env_info["act_clip"]:
                    action = jnp.clip(action, *action_clip)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                next_env_state = env.step(env_state, action)
                next_env_state = next_env_state.replace(
                    obs=normalize(next_env_state.obs, _normalizer_state)
                )

                transition = Transition(
                    prev_done=env_state.done,
                    done=next_env_state.done,
                    action=action,
                    prev_action=last_act,
                    value=value,
                    reward=next_env_state.reward,
                    prev_reward=env_state.reward,
                    log_prob=log_prob,
                    obs=env_state.obs,
                    # next_obs= # next_env_state.obs,
                    hidden=prev_hstate,
                    info=env_state.info,
                    # state=env_state.pipeline_state,
                )

                # Action fed to the Meta-Learner is one-hot encoded for discrete envs.
                re_action = (
                    jax.nn.one_hot(action, env.action_size) if _discrete else action
                )

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
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.collect_steps
            )
            train_state, env_state, _normalizer_state, re_action, hstate, rng = (
                runner_state
            )

            # UPDATE NORMALIZER
            if config.normalize_obs:
                _normalizer_state = running_statistics.update(
                    _normalizer_state, traj_batch.obs
                )

            # CALCULATE ADVANTAGES
            # Compute the last value
            x = normalize(env_state.obs, _normalizer_state)
            if config.meta_rl:
                x = jnp.concatenate([x, re_action, env_state.reward], axis=-1)
            ac_in = (x[None], env_state.done[None, :])
            _, _, _last_val = network.apply(
                train_state.params, hstate, ac_in, rngs={"default": rng}
            )

            gae, val = calculate_gae(
                traj_batch,
                traj_batch.value,
                _last_val[0],
                config.gamma,
                config.gae_lambda,
            )

            # Swap axes to make batch major
            batch_major = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), (traj_batch, gae, val)
            )
            # Add to buffer
            buffer_state = buffer.init(jax.tree.map(lambda x: x[0][0], batch_major))
            buffer_state = buffer.add(buffer_state, batch_major)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                (train_state, rng) = update_state

                def _update_minbatch(
                    train_state: TrainState,
                    batch: tuple[Transition, jax.Array, jax.Array],
                ):
                    def _loss_fn(params):
                        transition, _gae, _val = batch
                        _init_hstate = jax.tree.map(
                            lambda a: a[0], transition.hidden
                        )  # T=0, B, H
                        # RERUN NETWORK
                        x = normalize(transition.obs, _normalizer_state)
                        if config.meta_rl:
                            x = jnp.concatenate(
                                [
                                    x,
                                    transition.prev_action,
                                    transition.prev_reward.reshape((*x.shape[:2], 1)),
                                ],
                                axis=-1,
                            )
                        _, pi, v_hat = network.apply(
                            params,
                            _init_hstate,
                            (x, transition.prev_done),
                            rngs={"default": rng},
                        )
                        log_prob = pi.log_prob(transition.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = transition.value + (
                            v_hat - transition.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(_val - v_hat)
                        value_losses_clipped = jnp.square(_val - value_pred_clipped)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        diff = log_prob - transition.log_prob
                        if not env_info["discrete"]:
                            diff = diff.mean(axis=-1)
                        # diff = jnp.clip(diff, max=10)  # HACK avoids some NaNs!
                        ratio = jnp.exp(diff)
                        _gae = (_gae - _gae.mean()) / (_gae.std() + 1e-8)
                        _gae = _gae.reshape(*ratio.shape)
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

                        if hasattr(pi, "distribution"):
                            entropy = pi.distribution.entropy().mean()
                        else:
                            entropy = pi.entropy().mean()
                        if config.ent_coef:
                            total_loss -= entropy_schedule(epoch) * entropy

                        loss_info = {
                            "value_loss": value_loss,
                            "loss_actor": loss_actor,
                            "entropy": entropy,
                            "gae": _gae,
                            "log_prob_diff": diff,
                        }
                        if config.sparsity_penalty:
                            from jax_rtrl.models.regularization import (
                                sparsity_log_penalty,
                            )

                            sparsity_loss = sparsity_log_penalty(
                                {
                                    k: v["kernel"]
                                    for k, v in params["params"].items()
                                    if "actor" in k
                                }
                            )
                            loss_info["sparsity_loss"] = sparsity_loss
                            total_loss += config.sparsity_penalty * sparsity_loss

                        return total_loss, loss_info

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, loss_info), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss_info

                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng)
                experience = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(x, 0, 1), minibatch.experience
                )

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
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, np.arange(config.update_epochs)
            )
            runner_state = (
                update_state[0],
                env_state,
                _normalizer_state,
                re_action,
                hstate,
                update_state[-1],
            )
            for n in range(env.action_size):
                loss_info["action_%d_mean" % n] = re_action[:, n].mean()
                loss_info["action_%d_std" % n] = re_action[:, n].std()
            loss_info["train_reward"] = compute_agg_reward(traj_batch)
            return runner_state, loss_info

        runner_state = (
            train_state,
            env_state,
            normalizer_state,
            jnp.zeros((config.env_params.batch_size, env.action_size)),
            init_hstate,
            _rng,
        )

        # Get initial reward
        best_eval_reward, _traj = eval_model(runner_state[0].params, runner_state[2])

        steps_since_best = 0
        logger["initial_eval"] = best_eval_reward
        print(f"Initial eval reward: {best_eval_reward:.2f}")
        logger.log({"eval/rewards": best_eval_reward}, step=0)
        trajectories: Transition = None
        pbar = trange(config.episodes, desc="Training")
        try:
            for i in pbar:
                runner_state, loggables = jax.lax.scan(
                    update_step,
                    runner_state,
                    xs=i * config.update_steps + np.arange(config.update_steps),
                )
                if config.eval_every and (i % config.eval_every == 0):
                    eval_reward, _traj = eval_model(
                        runner_state[0].params, runner_state[2]
                    )

                    timestep = runner_state[0].step
                    loggables = {
                        **jax.tree.map(jnp.mean, loggables),
                        "eval/rewards": eval_reward,
                        "runner_step": timestep,
                        "dones": _traj.done.sum(),
                    }
                    # HACK: logging episode_reward to be consistent with RTRRL
                    loggables["episode_reward"] = loggables["train_reward"]

                    if config.log_norms:
                        loggables.update(**log_norms(runner_state[0].params)[0])
                    logger.flush()
                    new_best = False
                    if eval_reward > best_eval_reward:
                        steps_since_best = 0
                        new_best = True
                        best_eval_reward = logger["best_eval_reward"] = loggables[
                            "best_eval_reward"
                        ] = float(eval_reward)
                        trajectories = _traj
                        best_params = runner_state[0].params
                    else:
                        steps_since_best += 1
                    log_steps = (
                        (i + 1) * config.collect_steps * config.env_params.batch_size
                    )

                    logger.log(loggables, step=log_steps)
                    print(
                        f"Global step: {timestep:2.0e}, eval reward: {eval_reward:.2f}, best: {best_eval_reward:.2f}, ent: {loggables['entropy']:.2f}, train_reward: {loggables['train_reward']:.2f}"
                    )

                    # Render if we did better
                    should_render = (
                        config.render_every_evals is not None
                        and (
                            i % (config.eval_every * config.render_every_evals) == 0
                            and i > 0
                        )
                        or i == config.episodes - 1
                    )
                    if (
                        logger is not None
                        and config.render
                        and (new_best or should_render)
                    ):
                        pbar.write("Rendering env...")
                        frames = render_frames(
                            env,
                            _traj.env_state,
                            config.render_start,
                            config.render_start + config.render_steps,
                        )
                        if frames:
                            if env.name == "dronegym":
                                logger.log_img(
                                    "env",
                                    frames,
                                    step=log_steps,
                                    caption=f"Reward: {eval_reward:.2f}",
                                )
                                plt.close(frames)
                            else:
                                logger.log_video(
                                    "env/video",
                                    np.array(frames),
                                    step=log_steps,
                                    fps=30,
                                    caption=f"Reward: {eval_reward:.2f}",
                                )

                    # Early stopping
                    if config.patience and steps_since_best >= config.patience:
                        print(f"Early stopping patience {config.patience}")
                        break
        except KeyboardInterrupt:
            print("Interrupted by user, Finalizing...")
        finally:
            # Save the best model
            if config.save_model:
                save_model(best_params)
                logger.log_model("best_model", ckpt_path)
                print("Uploaded best model to wandb.")
            if config.record_best_eval_episode and trajectories is not None:

                def _prep_ep(t):
                    # Swap axes to batch major
                    t = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), t)

                    return {
                        "obs": t.obs,
                        "action": t.action,
                        "reward": t.reward,
                        "done": t.done,
                        **t.info,
                        **t.state,
                    }

                out_dir = f"data/{config.env_params.env_name}"
                os.makedirs(out_dir, exist_ok=True)
                # Save last episode data for plotting.
                np.savez(
                    f"{out_dir}/ppo_last_trajectory_{str(config.seed)}.npz",
                    **_prep_ep(_traj),
                )

                # Save best episode data for plotting.
                np.savez(
                    f"{out_dir}/ppo_best_trajectory_{str(config.seed)}.npz",
                    **_prep_ep(trajectories),
                )

                _prep_ep(trajectories)
                env_params = getattr(env, "params")
                if env_params:
                    with open(
                        f"{out_dir}/ppo_env_params_{str(config.seed)}.pkl", "wb"
                    ) as f:
                        pickle.dump(env_params, f)
        return eval_reward if config.eval_every else None

    return train


def train_and_eval(
    config: PPOParams,
    logger=DummyLogger(),
    param_overrides=None,
    network_cls=None,
    obs_transform_fn=None,
    act_transform_fn=None,
):
    """Run training."""
    rng = jax.random.PRNGKey(config.seed)
    logger["best_eval_reward"] = -np.inf
    try:
        result = make_train(
            config,
            logger,
            param_overrides=param_overrides,
            network_cls=network_cls,
            obs_transform_fn=obs_transform_fn,
            act_transform_fn=act_transform_fn,
        )(rng)

        if config.env_params.env_name == "dronegym":
            from jax_rl_util.envs.plot_drones import plot_from_file

            # CUSTOM Plotting
            out_dir = f"data/{config.env_params.env_name}"
            # Plot best trajectory
            plot_from_file(
                f"{out_dir}/ppo_best_trajectory_{str(config.seed)}.npz",
                f"{out_dir}/ppo_env_params_{str(config.seed)}.pkl",
                config.env_params.init_kwargs,
            )
            logger.log_img(
                "best_trajectories",
                plt.gcf(),
                caption="Total reward: {:.2f}".format(logger["best_eval_reward"]),
            )
            # Plot last trajectory
            plot_from_file(
                f"{out_dir}/ppo_last_trajectory_{str(config.seed)}.npz",
                f"{out_dir}/ppo_env_params_{str(config.seed)}.pkl",
                config.env_params.init_kwargs,
            )
            logger.log_img(
                "last_trajectories", plt.gcf(), caption=f"Total reward: {result:.2f}"
            )

        return logger["best_eval_reward"]
    except Exception as e:
        raise e
    finally:
        logger.finalize()


if __name__ == "__main__":
    hparams: PPOParams = simple_parsing.parse(PPOParams, add_config_path_arg=True)
    if hparams.ckpt_path and hparams.fresh:
        print(f"Restoring config from: {hparams.ckpt_path}")
        restored = restore_config(hparams.ckpt_path)
        if restored:
            restored = normalize_legacy_optimizer_config(restored)
            restored["restore_from"] = hparams.ckpt_path
            hparams = PPOParams(**restored)
    best_reward = with_logger(train_and_eval, hparams)
    print(f"Best eval reward: {best_reward:.2f}")
