"""RNN Actor-Critic flax Module."""

from dataclasses import field
from typing import Tuple

import distrax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from jax import random as jrandom
from jax_rtrl.models.jax_util import sigmoid_between
from jax_rtrl.models.mlp import MLP, FADense
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig


class AC(nn.Module):
    """TD lambda."""

    a_dim: int
    discrete: bool
    act_bounds: tuple[float, ...] | None = None
    act_log_bounds: tuple[float, ...] = field(default_factory=lambda: [0.01, 5])
    mlp_actor: bool = False
    f_align: bool = False
    action_noise: float = 0.0  # TODO: Implement action noise for exploration
    actor_layers: tuple[int, ...] = field(default_factory=lambda: [64])
    critic_layers: tuple[int, ...] = field(default_factory=lambda: [64])

    def setup(self) -> None:
        """Initialize components."""

        # Actor
        class Actor(nn.Module):
            layers: list[int]
            f_align: bool
            discrete: bool
            a_dim: int
            act_log_bounds: tuple[float, ...]
            act_bounds: tuple[float, ...] | None = None

            @nn.compact
            def __call__(self, hidden):
                """Compute action distribution form latent."""
                # actor_out_dim = self.a_dim if self.discrete else 2 * self.a_dim
                _actor = MLP(
                    self.layers + (self.a_dim,),
                    f_align=self.f_align,
                    name="mean",
                )

                if not self.discrete:
                    loc = _actor(hidden)
                    if len(loc.shape) > 1:
                        # Take mean of ... ensemble?
                        loc = loc.mean(axis=-2)
                    log_std = self.param(
                        "log_std", nn.initializers.zeros_init(), self.a_dim
                    )
                    # scale = sigmoid_between(scale, *self.act_log_bounds)
                    # scale = jnp.exp(scale) + self.act_log_bounds[0]
                    # scale = jax.nn.softplus(scale) + self.act_log_bounds[0]
                    if self.act_bounds is not None:
                        loc = sigmoid_between(loc, *self.act_bounds)
                    dist = distrax.LogStddevNormal(
                        loc,
                        log_std + self.act_log_bounds[0],
                        max_scale=self.act_log_bounds[1],
                    )
                else:
                    logits = _actor(hidden).mean(axis=-2)
                    dist = distrax.Categorical(logits=logits)
                return dist

        self.actor = Actor(
            self.actor_layers,
            self.f_align,
            self.discrete,
            self.a_dim,
            act_bounds=self.act_bounds,
            act_log_bounds=self.act_log_bounds,
            name="actor",
        )
        # Critic
        self.critic = MLP(
            self.critic_layers + (1,),
            f_align=self.f_align,
            # kernel_init=nn.initializers.zeros_init(),
            name="critic",
        )

    def loss(self, x, action, critic_weight: float = 1.0):
        """Compute loss."""
        critic_loss = self.critic(x).mean()
        dist = self.actor(x)
        actor_loss = dist.log_prob(action).mean()
        return actor_loss + critic_weight * critic_loss

    def value(self, x):
        """Compute value from latent."""
        return self.critic(x)

    def policy(self, x, sample_act: bool = False):
        """Compute action distribution form latent."""
        dist = self.actor(x)
        if sample_act:
            action = dist.sample(seed=self.make_rng("sampling"))
            if self.act_bounds is not None:
                action = jnp.clip(action, *self.act_bounds)
            return action, dist
        return dist

    def __call__(self, x):
        return self.actor(x), self.critic(x)


class RNNActorCritic(nn.RNNCellBase):
    """RTRRL cell with shared RNN and linear actor and critic networks."""

    a_dim: int
    discrete: bool
    obs_dim: int = None
    rnn_config: RNNEnsembleConfig = field(default_factory=RNNEnsembleConfig)
    f_align: bool = True
    act_log_bounds: tuple[float] = field(default_factory=lambda: [0.01, 1])
    shared: bool = False
    act_bounds: tuple[float] | None = None
    pass_obs: bool = False
    mlp_actor: bool = False
    pred_obs: bool = False
    layer_norm: bool = False

    @nn.nowrap
    def _make_rnn(self):
        if self.shared:
            return RNNEnsemble(self.rnn_config, name="rnn")
        else:
            if self.rnn_config.num_modules != 2:
                raise ValueError(
                    "RNNActorCritic num_modules has to be 2 when shared is False."
                )
            return RNNEnsemble(self.rnn_config, name="rnn")

    def setup(self) -> None:
        """Initialize components."""
        if self.rnn_config.model_name:
            self.rnn = self._make_rnn()

        # Make an ensemble of actor and critic using flax.linen.vmap
        # _vmap_td = nn.vmap(
        #     AC,
        #     variable_axes={"params": 0, "hidden": None, "falign": 0},
        #     split_rngs={"params": True, "falign": True},
        #     methods=["actor", "critic"],
        #     axis_size=self.num_modules,
        # )
        self.td = AC(
            self.a_dim,
            self.discrete,
            self.act_bounds,
            self.act_log_bounds,
            self.mlp_actor,
            self.f_align,
            name="td",
        )

        if self.pred_obs:
            self.obs = FADense(
                self.obs_dim + 1,  # Predict obs and reward
                f_align=self.f_align,
                #    kernel_init=nn.initializers.zeros_init(),
                #  use_bias=False,
                name="obs",
            )

        if self.layer_norm:
            self._layer_norm = nn.LayerNorm(use_bias=False, use_scale=False)

    def rnn_step(self, carry, obs, training=True, **kwargs):
        """Step RNN."""
        # Layer Norm
        if self.layer_norm:
            obs = self._layer_norm(obs)
        if not self.rnn_config.model_name:
            return obs, carry
        if carry is None:
            # Initialize seed and the carry
            carry = self.initialize_carry(jrandom.PRNGKey(self.random_seed), obs.shape)
        carry, hidden = self.rnn(carry, obs, **kwargs)
        # Layer Norm
        if self.layer_norm:
            hidden = self._layer_norm(hidden)
        return hidden, carry

    def value(self, hidden, x=None):
        """Compute value from latent."""
        if not self.shared:
            # hidden = jnp.concatenate([jax.lax.stop_gradient(hidden[0]), hidden[1]], axis=-1)
            hidden = hidden[..., 1:, :]
        if self.pass_obs:
            if len(x.shape) < len(hidden.shape):
                x = jnp.expand_dims(x, -2)
            hidden = jnp.concatenate([hidden, x], axis=-1)
        return self.td.critic(hidden)

    def obs_prediction(self, hidden, a, x=None):
        """Compute observation prediction from latent."""
        hidden = jnp.concatenate([hidden, a.reshape(*hidden.shape[:-1], -1)], axis=-1)
        if self.pass_obs:
            if len(x.shape) < len(hidden.shape):
                x = jnp.expand_dims(x, -2)
            hidden = jnp.concatenate([hidden, x], axis=-1)
        return self.obs(hidden)

    def policy(
        self,
        hidden,
        x=None,
        sample_act: bool = False,
        selected_act=None,
    ):
        """Compute action distribution form latent."""
        if not self.shared:
            # hidden = jnp.concatenate([hidden[0], jax.lax.stop_gradient(hidden[1])], axis=-1)
            hidden = hidden[..., :1, :]
        if self.pass_obs:
            if len(x.shape) < len(hidden.shape):
                x = jnp.expand_dims(x, -2)
            hidden = jnp.concatenate([hidden, x], axis=-1)
        dist = self.td.actor(hidden)
        if sample_act:
            action = dist.sample(seed=self.make_rng("sampling"))
            # if self.num_modules > 1:
            #     # Select action corresponding to the module predicting the highest value
            #     batch_shape = () if action.ndim == 1 else action.shape[-2]
            #     if selected_act is None:
            #         selected_act = jrandom.randint(self.make_rng("sampling"), batch_shape, 0, self.num_modules)
            #     if action.ndim > 1:
            #         action = action[..., jnp.arange(batch_shape), selected_act]
            #     else:
            #         action = action[selected_act]
            if self.act_bounds is not None:
                action = jnp.clip(action, *self.act_bounds)
            return action, dist
        return dist

    @nn.compact
    def __call__(self, carry, x):
        """Step RNN and compute actor and critic."""
        # RNN
        hidden, new_carry = self.rnn_step(carry, x)

        # Critic
        v_hat = self.value(hidden, x)

        # selected_act = v_hat.argmax()

        # Actor
        action, _ = self.policy(hidden, x, True)

        if self.pred_obs:
            prediction = self.obs_prediction(hidden, action, x)
            out = (action, v_hat, prediction, hidden)
        else:
            out = (action, v_hat, hidden)
        return new_carry, out

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the Worldmodel cell carry."""
        if not self.rnn_config.model_name:
            return None

        return self._make_rnn().initialize_carry(rng, input_shape)
