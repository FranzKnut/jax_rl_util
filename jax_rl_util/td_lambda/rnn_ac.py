"""RNN Actor-Critic flax Module."""

from dataclasses import field

import distrax
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from jax_rtrl.networks.autoencoders import ConvEncoder, ConvConfig
from jax_rtrl.models.feedforward import DistributionLayer
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig
from jax_rtrl.networks.policies import PolicyConfig, PolicyRNN


Actor = DistributionLayer


class Critic(DistributionLayer):
    """Value critic head. Same as DistributionLayer with out_size=1 default."""

    out_size: int = 1
    distribution: str = "Deterministic"


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

        self.actor = PolicyRNN(
            self.a_dim,
            self.policy_config,
            # TODO: this config for splitting inputs to ensemble modules is confusing
            split_input=self.use_shared_rnn
            and (self.rnn_config.num_modules == self.policy_config.num_modules),
            name="actor",
        )
        self.critic = RNNEnsemble(
            self.critic_config,
            out_size=1,
            split_input=self.use_shared_rnn
            and (self.rnn_config.num_modules == self.critic_config.num_modules),
            name="critic",
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
                self.policy_config.latent_size,
                self.cnn_config,
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
        return self.critic(v_hidden, encoded, training=training)

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
    ) -> tuple[jax.Array, distrax.Distribution] | jax.Array:
        # if not self.shared:
        #     # hidden = jnp.concatenate([hidden[0], jax.lax.stop_gradient(hidden[1])], axis=-1)
        #     hidden = hidden[..., :1, :]
        # if self.pass_obs:
        #     if len(x.shape) < len(hidden.shape):
        #         x = jnp.expand_dims(x, -2)
        #     hidden = jnp.concatenate([hidden, x], axis=-1)
        encoded, (combined_dist, dists) = self.actor(
            pi_state, encoded, img, training=training
        )
        if sample_act:
            greedy_action = dists.mode()
            if not training:
                action = greedy_action
            else:
                _rng = jax.random.uniform(
                    self.make_rng("default"), shape=greedy_action.shape
                )
                # Epsilon-greedy action selection
                action = jnp.where(
                    _rng < epsilon,
                    greedy_action,
                    combined_dist.sample(seed=self.make_rng("default")),
                )
            if self.act_bounds is not None:
                action = jnp.clip(action, *self.act_bounds)
            return encoded, (action, combined_dist)
        return encoded, combined_dist

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
            input_shape = input_shape[:-3] + (self.policy_config.latent_size,)

        return self.rnn.initialize_carry(rng, input_shape)

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the submodule states."""

        if self.use_cnn:
            input_shape = input_shape[:-3] + (self.policy_config.latent_size,)
        if self.rnn_config is not None and self.rnn_config.model_name:
            rnn_state = self._init_shared_rnn(rng, input_shape)
            input_shape = input_shape[:-1] + self.rnn_config.layers[-1:]
        else:
            rnn_state = None
        v_state = self.critic.initialize_carry(rng, input_shape)
        pi_state = self.actor.initialize_carry(rng, input_shape)
        return (rnn_state, v_state, pi_state)
