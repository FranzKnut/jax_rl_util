"""RNN Actor-Critic flax Module."""

from dataclasses import field
from numbers import Number

import distrax
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from jax_rtrl.models.jax_util import get_normalization_fn, sigmoid_between
from jax_rtrl.models.feedforward import MLP, FADense
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig


# Actor
class Actor(nn.Module):
    layers: list[int]
    f_align: bool
    discrete: bool
    a_dim: int
    act_log_bounds: float | tuple[float, float] | None = -1
    act_bounds: tuple[float, float] | None = None
    act_dist_name: str = "normal"
    norm: str | None = None  # Normalization type, e.g., "layer", "batch"

    @nn.compact
    def __call__(self, hidden, training=True):
        """Compute action distribution from latent."""
        # actor_out_dim = self.a_dim if self.discrete else 2 * self.a_dim
        if self.layers:
            hidden = MLP(
                self.layers,
                f_align=self.f_align,
                name="mean",
                norm=self.norm,
            )(hidden)
        hidden = get_normalization_fn(self.norm, training=training)(hidden)
        model_out = FADense(
            self.a_dim * 2
            if self.act_dist_name in ["beta", "brax", "normal_scale"]
            else self.a_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)

        if self.discrete:
            # logits = model_out.mean(axis=-2)
            dist = distrax.Categorical(logits=model_out)
        elif self.act_dist_name == "deterministic":
            # Deterministic action, no distribution
            # model_out = model_out.mean(axis=-2)
            if self.act_bounds is not None:
                model_out = sigmoid_between(model_out, *self.act_bounds)
            dist = distrax.Deterministic(model_out)
        else:
            if self.act_dist_name == "beta":
                if self.act_bounds is not None:
                    # If action limits are defined we sample from [0, 1] and transform the event.
                    act_range = jnp.array(self.act_bounds[1]) - jnp.array(
                        self.act_bounds[0]
                    )
                    act_min = jnp.array(self.act_bounds[0])
                    scaling_transform = distrax.ScalarAffine(act_min, act_range)
                alpha = jax.nn.softplus(model_out[..., : model_out.shape[-1] // 2])
                beta = jax.nn.softplus(model_out[..., model_out.shape[-1] // 2 :])
                return distrax.Transformed(distrax.Beta(alpha, beta), scaling_transform)
            elif self.act_dist_name == "brax":
                from brax.training.distribution import NormalTanhDistribution

                return NormalTanhDistribution(
                    event_size=self.a_dim,
                    min_std=jnp.exp(self.act_log_bounds),
                ).create_dist(model_out)
            else:
                if self.act_dist_name == "normal_scale":
                    loc, log_std = jnp.split(model_out, 2, axis=-1)
                else:
                    loc = model_out
                    log_std = self.param(
                        "log_std", nn.initializers.zeros_init(), self.a_dim
                    )
                # if len(loc.shape) > 1:
                #     # Take mean of ... ensemble?
                #     loc = loc.mean(axis=-2)
                #     if self.act_dist_name == "normal_scale":
                #         log_std = log_std.mean(axis=-2)

                if isinstance(self.act_log_bounds, tuple):
                    log_std = sigmoid_between(log_std, *self.act_log_bounds)
                elif isinstance(self.act_log_bounds, Number):
                    log_std = jax.nn.softplus(log_std) + self.act_log_bounds
                if self.act_bounds is not None:
                    loc = sigmoid_between(loc, *self.act_bounds)
                dist = distrax.LogStddevNormal(
                    loc,
                    log_std,
                    # max_scale=self.act_log_bounds[1],
                )

        return dist


class Critic(nn.Module):
    """Critic network."""

    layers: list[int] = field(default_factory=list)
    f_align: bool = False
    norm: str | None = None  # Normalization type, e.g., "layer", "batch"

    @nn.compact
    def __call__(self, x, training=True):
        """Compute value from latent."""
        if self.layers:
            x = MLP(
                self.layers,
                f_align=self.f_align,
                name="mlp",
                norm=self.norm,
            )(x)
        x = get_normalization_fn(self.norm, training=training)(x)

        return FADense(
            1,
            # kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="critic_head",
        )(x)


class AC(nn.Module):
    """TD lambda."""

    a_dim: int
    discrete: bool
    split_actor: bool = False
    act_bounds: tuple[float, ...] | None = None
    act_log_bounds: tuple[float, float] | float | None = None
    act_dist_name: str = "normal"
    actor_layers: tuple[int, ...] = ()
    critic_layers: tuple[int, ...] = ()
    f_align: bool = False
    norm: str | None = None  # Normalization type, e.g., "layer", "batch"

    # action_noise: float = 0.0  # TODO: Implement action noise for exploration

    def setup(self) -> None:
        """Initialize components."""
        # Actor
        self.actor = Actor(
            self.actor_layers,
            self.f_align,
            self.discrete,
            self.a_dim,
            act_bounds=self.act_bounds,
            act_log_bounds=self.act_log_bounds,
            act_dist_name=self.act_dist_name,
            norm=self.norm,
            name="actor",
        )
        # Critic
        self.critic = Critic(
            self.critic_layers,
            self.f_align,
            norm=self.norm,
            name="critic",
        )

    def value(self, x, training: bool = True):
        """Compute value from latent."""
        if not self.split_actor and x.shape[-2] > 1:
            # First module of the ensemble is used for the actor
            x = x[..., 1:, :]  # Assume first axis is ensemble axis
        return self.critic(x, training=training)

    def policy(self, x, sample_act: bool = False, training: bool = True):
        """Compute action distribution or sample actions from the policy network.

        Args:
            x: Latent representation or input features for the policy network.
            sample_act (bool, optional): If True, returns sampled actions along with
                the distribution. If False, returns only the distribution. Defaults to False.
            deterministic (bool, optional): If True and sample_act is True, returns
                the mode of the distribution (deterministic action). If False, samples
                stochastically. Only applies when sample_act is True. Defaults to False.

        Returns:
            If sample_act is False:
                Distribution: The action distribution from the actor network.
            If sample_act is True:
                tuple: A tuple containing:
                    - action: Sampled action (clipped to action bounds if specified)
                    - dist: The action distribution from the actor network

        Notes:
            - When sample_act is True and self.act_bounds is defined,
                actions are automatically clipped to action bounds.
            - Uses internal RNG state for stochastic sampling via self.make_rng("sampling").
        """
        if not self.split_actor:
            # First module of the ensemble is used for the actor
            x = x[..., 0, :]  # Assume first axis is ensemble axis
        dist = self.actor(x, training=training)
        if sample_act:
            if not training:
                action = dist.mode()
            else:
                action = dist.sample(seed=self.make_rng("sampling"))
            if self.act_bounds is not None:
                action = jnp.clip(action, *self.act_bounds)
            return action, dist
        return dist

    def __call__(self, x, sample_act: bool = False, training: bool = True):
        return self.policy(x, sample_act, training), self.value(x)

    @nn.nowrap
    def loss(
        self,
        params,
        x,
        action=None,
        critic_weight: float = 1.0,
        entropy_weight: float = 0.0,
        training: bool = True,
    ):
        """Compute loss. Also returns sampled action if action is not provided.

        FIXME: This is not actually a loss since it should be maximized.
        """
        sample_act = action is None
        dist, value = self.apply(
            params,
            x,
            sample_act=sample_act,
            training=training,
        )
        if sample_act:
            action, dist = dist
        critic_loss = value.mean()
        actor_loss = dist.log_prob(action).mean()
        # Add entropy to the actor loss
        entropy = dist.entropy().mean()
        total_loss = actor_loss + critic_weight * critic_loss - entropy_weight * entropy
        info = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
            "ac_total_loss": total_loss,
        }
        aux = (value, action, info) if sample_act else (value, info)
        return total_loss, aux


class RNNActorCritic(nn.RNNCellBase):
    """RTRRL cell with shared RNN and linear actor and critic networks."""

    a_dim: int
    discrete: bool
    obs_dim: int = None
    rnn_config: RNNEnsembleConfig = field(default_factory=RNNEnsembleConfig)
    split_actor: bool = False
    f_align: bool = True
    act_log_bounds: tuple[float, float] | float | None = -1
    shared: bool = False
    act_bounds: tuple[float] | None = None
    act_dist_name: str = "normal"
    pass_obs: bool = False
    actor_layers: tuple[int, ...] = ()
    critic_layers: tuple[int, ...] = ()
    pred_obs: bool = False
    layer_norm: bool = False

    def setup(self) -> None:
        """Initialize components."""
        if self.rnn_config.model_name:
            if self.shared:
                self.rnn = RNNEnsemble(self.rnn_config, name="rnn")
            else:
                if self.rnn_config.num_modules != 2:
                    raise ValueError(
                        "RNNActorCritic num_modules has to be 2 when shared is False."
                    )
                self.rnn = RNNEnsemble(self.rnn_config, name="rnn")

        # Make an ensemble of actor and critic using flax.linen.vmap
        # _vmap_td = nn.vmap(
        #     AC,
        #     variable_axes={"params": 0, "hidden": None, "falign": 0},
        #     split_rngs={"params": True, "falign": True},
        #     methods=["actor", "critic"],
        #     axis_size=self.num_modules,
        # )
        self.ac = AC(
            a_dim=self.a_dim,
            discrete=self.discrete,
            split_actor=self.split_actor,
            act_bounds=self.act_bounds,
            act_log_bounds=self.act_log_bounds,
            actor_layers=self.actor_layers,
            critic_layers=self.critic_layers,
            f_align=self.f_align,
            act_dist_name=self.act_dist_name,
            name="ac",
        )

        if self.pred_obs:
            self.obs = FADense(
                self.obs_dim + 1,  # Predict obs and reward
                f_align=self.f_align,
                #    kernel_init=nn.initializers.zeros_init(),
                #  use_bias=False,
                name="obs",
            )

    def rnn_step(self, carry, obs, training=True, **kwargs):
        """Step RNN."""
        if not self.rnn_config.model_name:
            return obs, carry
        if carry is None:
            # Initialize seed and the carry
            carry = self.initialize_carry(self.make_rng(), obs.shape)
        carry, hidden = self.rnn(carry, obs, training, **kwargs)
        return hidden, carry

    def value(self, hidden, x=None, training=True):
        """Compute value from latent."""
        if not self.shared:
            # hidden = jnp.concatenate([jax.lax.stop_gradient(hidden[0]), hidden[1]], axis=-1)
            hidden = hidden[..., 1:, :]
        if self.pass_obs:
            if len(x.shape) < len(hidden.shape):
                x = jnp.expand_dims(x, -2)
            hidden = jnp.concatenate([hidden, x], axis=-1)
        return self.ac.value(hidden, training=training)

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
        training: bool = True,
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
        return self.ac.policy(hidden, sample_act=sample_act, training=training)

    @nn.compact
    def __call__(self, carry, x, training=True):
        """Step RNN and compute actor and critic."""
        # RNN
        hidden, new_carry = self.rnn_step(carry, x, training=training)

        # Critic
        v_hat = self.value(hidden, x, training=training)

        # selected_act = v_hat.argmax()

        # Actor
        action, _ = self.policy(hidden, x, True, training=training)

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

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the Worldmodel cell carry."""
        if not self.rnn_config.model_name:
            return None

        return self.rnn.initialize_carry(rng, input_shape)
