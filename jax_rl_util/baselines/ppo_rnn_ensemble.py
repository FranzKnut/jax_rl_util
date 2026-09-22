"""PPO implementation that uses jax_rtrl RNNEnsemble as recurrent module."""

from dataclasses import dataclass, field, replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import simple_parsing
from jax_rl_util.baselines import ppo_rnn as base_ppo
from jax_rl_util.envs import EnvironmentConfig
from jax_rl_util.envs.wrappers import Env
from jax_rl_util.optimizers import OptimizerConfig
from jax_rl_util.td_lambda.rnn_ac import RNNActorCritic
from jax_rl_util.util.logging_util import DummyLogger, with_logger
from jax_rtrl.models.cells import CELL_TYPES
from jax_rtrl.networks.policies import PolicyConfig

RESET_AWARE_MODELS = {
    "lru",
    "lru_rtrl",
    "s5",
    "s5_rtrl",
}

MODEL_DOWNCAST = {
    "rflo": "bptt",
    "rtrl": "bptt",
    "eprop": "bptt",
    "snap0": "bptt",
    "ltc_rtrl": "ltc",
    "ltc_rflo": "ltc",
    "ltc_snap0": "ltc",
    "lrc_rtrl": "lrc",
    "lrc_snap0": "lrc",
    "lru_rtrl": "lru",
    "s5_rtrl": "s5",
}


@dataclass
class PPOEnsembleParams(base_ppo.PPOParams):
    """PPO params with RNNEnsemble backbone configuration."""

    model_config: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(
            model_name="bptt",
            hidden_size=32,
            out_dist="NormalTanh",
            dist_scale_bounds=(1e-5, 1.0),
        )
    )
    train_encoder: bool = False
    env_params: EnvironmentConfig = field(
        default_factory=lambda: EnvironmentConfig(
            env_name="halfcheetah",
            init_kwargs={"backend": "mjx"},
            batch_size=32,
        )
    )
    optimizer_params: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            opt_name="adam",
            learning_rate=1e-3,
        )
    )


def _resolve_rnn_config(config: "PPOEnsembleParams") -> PolicyConfig:
    """Resolve final RNNEnsemble config from PPO and optional RTRRL-style fields."""
    base_cfg = config.model_config
    model_name = base_cfg.model_name
    if model_name.lower() in CELL_TYPES:
        mapped = MODEL_DOWNCAST.get(model_name, model_name)
        if mapped != model_name:
            print(f"WARNING: PPOEnsemble uses '{mapped}' instead of online model '{model_name}'.")
        model_name = mapped
    updates = {"model_name": model_name}
    if model_name is not None and base_cfg.hidden_size is None and base_cfg.layers is None:
        updates["hidden_size"] = config.num_units

    return replace(base_cfg, **updates)


def _extract_value(value_dist):
    if isinstance(value_dist, tuple):
        value_dist = value_dist[0]
    if hasattr(value_dist, "mode"):
        return value_dist.mode()
    if hasattr(value_dist, "mean"):
        return value_dist.mean()
    return value_dist


class EnsembleActorCritic(nn.Module):
    """Actor-critic wrapper that uses RNNActorCritic with RNNEnsemble configs."""

    action_dim: int
    discrete: bool
    config: "PPOEnsembleParams"
    action_limits: jnp.ndarray | None = None

    def setup(self):
        """Initialize the RNNActorCritic cell with ensemble configs."""
        policy_cfg = _resolve_rnn_config(self.config)
        use_cnn = bool(policy_cfg.use_cnn)
        if use_cnn:
            policy_cfg = replace(policy_cfg, use_cnn=False)
        if self.discrete:
            policy_cfg = replace(policy_cfg, out_dist="Categorical")
        elif policy_cfg.out_dist in {None, "Deterministic"}:
            policy_cfg = replace(policy_cfg, out_dist="LogStddevNormal")

        critic_cfg = replace(policy_cfg, out_dist="Deterministic")

        self.cell = RNNActorCritic(
            a_dim=self.action_dim,
            discrete=self.discrete,
            # rnn_config=policy_cfg,
            policy_config=policy_cfg,
            critic_config=critic_cfg,
            use_cnn=use_cnn,
            cnn_config=policy_cfg.cnn_config,
            act_bounds=self.action_limits,
        )

    @nn.compact
    def __call__(self, carry, x):
        """Run the recurrent actor-critic over a time-major sequence."""
        obs, dones = x
        h0 = self.initialize_carry(self.make_rng("default"), obs.shape[1:])
        if carry is None:
            carry = h0

        # Pre-initialize variables outside scan to prevent data dependency tracer leaks
        _c_dummy = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), h0)
        _obs_dummy = jnp.zeros(obs.shape[2:], obs.dtype)
        _done_dummy = jnp.zeros((), dones.dtype)
        self.cell(_c_dummy, _obs_dummy, reset=_done_dummy, training=True)
        _train_encoder = self.config.train_encoder

        class _StepModule(nn.Module):
            cell: RNNActorCritic

            @nn.compact
            def __call__(self, carry_i, xs):
                obs_i, done_i, h0_i = xs
                carry_i = jax.tree.map(lambda a, b: jnp.where(done_i, a, b), h0_i, carry_i)
                encoded, rnn_state = self.cell.encode(carry_i[0], obs_i, reset=done_i, training=True)
                if not _train_encoder:
                    encoded = jax.lax.stop_gradient(encoded)
                    rnn_state = jax.lax.stop_gradient(rnn_state)
                v_state, v_dist = self.cell.value(encoded, obs_i, carry_i[1], training=True)
                pi_state, _pi_dist = self.cell.policy(encoded, pi_state=carry_i[2], training=True)

                _value = _extract_value(v_dist)
                if isinstance(_pi_dist, tuple):
                    _pi_dist = _pi_dist[0]
                return (rnn_state, v_state, pi_state), (_pi_dist, _value)

        batch_step = nn.vmap(
            _StepModule,
            variable_axes={k: None for k in ["params", "falign", "batch_stats", "wiring"]},
            split_rngs={k: False for k in ["params", "falign", "batch_stats", "wiring"]},
            in_axes=(0, 0),
            out_axes=(0, 0),
        )

        scan_step = nn.scan(
            batch_step,
            variable_broadcast=["params", "falign", "batch_stats", "wiring"],
            split_rngs={k: False for k in ["params", "falign", "batch_stats", "wiring"]},
            in_axes=0,
            out_axes=0,
        )

        h0_t = jax.tree.map(lambda x: jnp.broadcast_to(x, (obs.shape[0], *x.shape)), h0)

        carry, (pi_dist, value) = scan_step(cell=self.cell, name="scan_step")(carry, (obs, dones, h0_t))
        return carry, pi_dist, value

    @nn.nowrap
    def initialize_carry(self, rng, input_shape):
        """Initialize the recurrent carry state."""
        return self.cell.initialize_carry(rng, input_shape)


def _prepare_config(config: PPOEnsembleParams) -> PPOEnsembleParams:
    rnn_config = _resolve_rnn_config(config)
    updates = {
        "model_config": rnn_config,
    }
    if rnn_config.hidden_size is not None and rnn_config.hidden_size != config.num_units:
        updates["num_units"] = rnn_config.hidden_size
    return replace(config, **updates)


def train_and_eval(
    config: PPOEnsembleParams,
    logger=DummyLogger(),
    param_overrides=None,
    env: Env = None,
):
    """Train PPO with RNNEnsemble dynamics."""
    config = _prepare_config(config)
    return base_ppo.train_and_eval(
        config,
        logger=logger,
        param_overrides=param_overrides,
        network_cls=EnsembleActorCritic,
        env=env,
    )


if __name__ == "__main__":
    args: PPOEnsembleParams = simple_parsing.parse(PPOEnsembleParams, config_path='config/ppo_rnn_ensemble.yaml', add_config_path_arg=True)
    if args.ckpt_path and args.fresh:
        restore_path = base_ppo.resolve_restore_path(args.ckpt_path)
        print(f"Restoring config from: {restore_path}")
        restored = base_ppo.restore_config(restore_path)
        if restored:
            restored = base_ppo.normalize_legacy_optimizer_config(restored)
            restored["ckpt_path"] = args.ckpt_path
            args = PPOEnsembleParams(**restored)
    best_reward = with_logger(train_and_eval, args)
    print(f"Best eval reward: {best_reward:.2f}")
