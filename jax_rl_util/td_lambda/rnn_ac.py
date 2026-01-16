"""RNN Actor-Critic flax Module."""

from dataclasses import field
from numbers import Number

import distrax
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from jax_rtrl.networks.autoencoders import ConvEncoder, ConvConfig
from jax_rtrl.util.jax_util import get_normalization_fn, sigmoid_between
from jax_rtrl.models.feedforward import MLP, FADense
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig
from jax_rtrl.networks.policies import PolicyConfig, PolicyRNN


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
            # kernel_init=nn.initializers.zeros_init(),
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
                dist = distrax.LogStddevNormal(loc, log_std)

        return dist


class AC(nn.Module):
    """TD lambda.

    TODO: Remove this subclass and just use RNNActorCritic directly.
    """

    a_dim: int
    discrete: bool
    policy_config: PolicyConfig
    critic_config: RNNEnsembleConfig
    # split_actor: bool = False
    act_bounds: tuple[float, ...] | None = None
    # TODO: this config for splitting inputs to ensemble modules is confusing
    split_critic_inputs: bool = False
    split_actor_inputs: bool = False
    # action_noise: float = 0.0  # TODO: Implement action noise for exploration

    def setup(self) -> None:
        """Initialize components."""
        # Actor
        self.actor = PolicyRNN(
            self.a_dim,
            self.policy_config,
            split_input=self.split_actor_inputs,
            name="actor",
        )
        # Critic
        self.critic = RNNEnsemble(
            self.critic_config,
            out_size=1,
            split_input=self.split_critic_inputs,
            name="critic",
        )

    def value(self, x, h=None, training: bool = True):
        """Compute value from latent."""
        # if not self.split_actor and x.ndim > 1 and x.shape[-2] > 1:
        #     # First module of the ensemble is used for the actor
        #     x = x[..., 1:, :]  # Assume first axis is ensemble axis
        return self.critic(h, x, training=training)

    def policy(
        self,
        encoded=None,
        img=None,
        pi_state=None,
        sample_act: bool = False,
        training: bool = True,
        greedy_epsilon: float = 0.0,
    ):
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
            - Uses internal RNG state for stochastic sampling via self.make_rng("default").
        """
        # if not self.split_actor and x.ndim > 1:
        #     # First module of the ensemble is used for the actor
        #     x = x[..., 0, :]  # Assume first axis is ensemble axis
        encoded, (combined_dist, dists) = self.actor(
            pi_state, encoded, img, training=training
        )
        if sample_act:
            greedy_action = dists.mode()
            if not training:
                action = greedy_action
            else:
                _rnd = jax.random.uniform(
                    self.make_rng("default"), shape=greedy_action.shape
                )
                # Epsilon-greedy action selection
                action = jnp.where(
                    _rnd < greedy_epsilon,
                    greedy_action,
                    combined_dist.sample(seed=self.make_rng("default")),
                )
            if self.act_bounds is not None:
                action = jnp.clip(action, *self.act_bounds)
            return encoded, (action, combined_dist)
        return encoded, combined_dist

    def __call__(
        self, x, sample_act: bool = False, training: bool = True, epsilon: float = 0.0
    ):
        return self.policy(
            x,
            sample_act,
            training,
            greedy_epsilon=epsilon,
        ), self.value(x)


class RNNActorCritic(nn.RNNCellBase):
    """RTRRL cell with shared RNN and linear actor and critic networks."""

    a_dim: int
    discrete: bool
    # obs_dim: int = None
    # img_dim: int = None
    rnn_config: RNNEnsembleConfig | None = field(default_factory=RNNEnsembleConfig)
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    critic_config: RNNEnsembleConfig = field(default_factory=RNNEnsembleConfig)
    use_cnn: bool = False
    cnn_config: ConvConfig = field(default_factory=ConvConfig)
    # act_log_bounds: tuple[float, float] | float | None = -1
    act_bounds: tuple[float] | None = None
    pass_obs: bool = False
    pred_obs: bool = False

    @property
    def use_shared_rnn(self) -> bool:
        """Whether to use a shared RNN for actor and critic."""
        return self.rnn_config is not None and self.rnn_config.model_name is not None

    def setup(self) -> None:
        """Initialize components."""
        if self.rnn_config is not None and self.rnn_config.model_name:
            self.rnn = RNNEnsemble(self.rnn_config, out_size=None, name="rnn")

        self.ac = AC(
            name="ac",
            a_dim=self.a_dim,
            policy_config=self.policy_config,
            critic_config=self.critic_config,
            discrete=self.discrete,
            act_bounds=self.act_bounds,
            # TODO: this config for splitting inputs to ensemble modules is confusing
            split_actor_inputs=self.use_shared_rnn
            and (self.rnn_config.num_modules == self.policy_config.num_modules),
            split_critic_inputs=self.use_shared_rnn
            and (self.rnn_config.num_modules == self.critic_config.num_modules),
        )

        if self.pred_obs:
            raise NotImplementedError("Observation prediction not implemented yet.")
            self.obs = RNNEnsemble(
                self.obs_dim + 1,  # Predict obs and reward
                # f_align=self.f_align,
                #    kernel_init=nn.initializers.zeros_init(),
                #  use_bias=False,
                name="obs",
            )

        if self.use_cnn:
            self.enc = ConvEncoder(
                latent_size=self.policy_config.latent_size,
                config=self.cnn_config,
                name="enc",
            )

    def encode(self, carry, obs, reset=False, img=None, training=True, **kwargs):
        """Step RNN."""

        if self.use_cnn:
            if img is None:
                img = obs
                obs = None
            encoded_img = self.enc(img)
            # If given, concatenate encoded image with vector observations
            if obs is not None:
                obs = jnp.concatenate([obs, encoded_img], axis=-1)
            else:
                obs = encoded_img

        if not self.use_shared_rnn:
            return obs, carry

        if self.rnn_config is not None and not self.rnn_config.model_name:
            # No RNN, just return repeated obs as hidden state
            obs = obs[None] * jnp.ones(
                (self.rnn_config.num_modules,) + (1,) * (obs.ndim)
            )
            return obs, carry

        carry, hidden = self.rnn(carry, obs, training, **kwargs)
        return hidden, carry

    def value(self, encoded, x=None, v_hidden=None, training=True):
        """Compute value from latent."""
        # if not self.shared:
        #     # hidden = jnp.concatenate([jax.lax.stop_gradient(hidden[0]), hidden[1]], axis=-1)
        #     hidden = hidden[..., 1:, :]
        if self.pass_obs:
            if len(x.shape) < len(encoded.shape):
                x = jnp.expand_dims(x, -2)
            encoded = jnp.concatenate([encoded, x], axis=-1)
        return self.ac.value(encoded, v_hidden, training=training)

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
        encoded,
        img=None,
        pi_state=None,
        sample_act: bool = False,
        training: bool = True,
        selected_act=None,
        epsilon: float = 0.0,
    ):
        """Compute action distribution form latent."""
        # if not self.shared:
        #     # hidden = jnp.concatenate([hidden[0], jax.lax.stop_gradient(hidden[1])], axis=-1)
        #     hidden = hidden[..., :1, :]
        # if self.pass_obs:
        #     if len(x.shape) < len(hidden.shape):
        #         x = jnp.expand_dims(x, -2)
        #     hidden = jnp.concatenate([hidden, x], axis=-1)
        return self.ac.policy(
            encoded,
            img,
            pi_state=pi_state,
            sample_act=sample_act,
            training=training,
            greedy_epsilon=epsilon,
        )

    @nn.compact
    def __call__(
        self, carry, x, img=None, reset=False, training=True, epsilon: float = 0.0
    ):
        """Step RNN and compute actor and critic."""
        h0 = self.initialize_carry(self.make_rng("default"), x.shape)
        if carry is None:
            # Initialize seed and the carry
            carry = h0
        else:
            carry = jax.tree.map(lambda a, b: jnp.where(reset, a, b), h0, carry)

        # RNN
        encoded, rnn_state = self.encode(
            carry[0], x, reset=reset, img=img, training=training
        )

        # Critic
        v_state, v_hat = self.value(encoded, x, carry[1], training=training)
        # selected_act = v_hat.argmax()

        # Actor
        pi_state, (action, _) = self.policy(
            encoded, img, carry[2], True, training=training, epsilon=epsilon
        )

        if self.pred_obs:
            prediction = self.obs_prediction(encoded, action, img)
            out = (action, v_hat, prediction, encoded)
        else:
            out = (action, v_hat, encoded)
        return (rnn_state, v_state, pi_state), out

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    def _init_shared_rnn(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the shared RNN cell."""
        if self.use_cnn:
            input_shape = input_shape[:-3] + (self.cnn_config.latent_size,)

        return self.rnn.initialize_carry(rng, input_shape)

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the submodule states."""

        if self.use_cnn:
            input_shape = input_shape[:-3] + (self.cnn_config.latent_size,)
        if self.rnn_config is not None and self.rnn_config.model_name:
            rnn_state = self._init_shared_rnn(rng, input_shape)
            input_shape = self.rnn_config.layers[-1:]
        else:
            rnn_state = None
        v_state = self.ac.critic.initialize_carry(rng, input_shape)
        pi_state = self.ac.actor.initialize_carry(rng, input_shape)
        return (rnn_state, v_state, pi_state)
