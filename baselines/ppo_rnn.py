"""PPO implementation in JAX."""

from dataclasses import dataclass, field
import json
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools
import optax

import warnings
from jax_rtrl.logging_util import DummyLogger, with_logger
from envs.environments import EnvironmentParams, make_env

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class PPO_Params:
    """Parameters for PPO."""

    debug: int = 0
    seed: int = 123
    episodes: int = 3000000
    patience: int = 20
    eval_every: int = 100
    eval_steps: int = 10000
    gamma: float = 0.99
    LR: float = 1e-5
    update_steps: int = 128
    rollout_steps: int = 32
    NUM_UNITS: int = 32
    UPDATE_EPOCHS: int = 4
    GAE_LAMBDA: float = 0.9
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.001
    VF_COEF: float = 0.5
    gradient_clip: float = 1
    dt: float = 1
    ANNEAL_LR: bool = True
    MODEL: str = "CTRNN"
    log_code: bool = False
    env_params: EnvironmentParams = field(
        default_factory=lambda: EnvironmentParams(env_name="StatelessCartPoleEasy", batch_size=4)
    )


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
            lambda new, old: jnp.where(resets[:, np.newaxis], new, old),
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

    config: Dict

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
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            h,
        )
        dense = nn.Dense(self.config.NUM_UNITS)
        tau = self.param("tau", nn.initializers.ones, (self.config.NUM_UNITS,))

        y = jnp.concatenate([ins, h], axis=-1)
        u = dense(y)
        act = jnp.tanh(u)
        dh = (act - h) / tau
        out = jax.tree.map(lambda a, b: a + b * self.config.dt, h, dh)
        return out, out

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        """See flax dokumantation for more info."""
        return jnp.zeros((batch_dims, hidden_size))


class GRU(nn.Module):
    """GRU module."""

    config: Dict

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
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(carry[0].shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        """Use a dummy key since the default state init fn is just zeros."""
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    """Actor Critic RNN."""

    action_dim: Sequence[int]
    config: Dict

    def dist(self, model_out, log_std_min=-5, log_std_max=None):
        """Split the output of the actor into mean and std.

        Applies squashing to the normal distribution
        Args:
            a (jnp.ndarray): Output of the actor

        Returns:
            jnp.ndarray: Mean of the distribution
            jnp.ndarray: Std of the distribution
        """
        if self.config.discrete:
            return distrax.Categorical(logits=model_out)
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
        action_dim = self.action_dim if self.config.discrete else self.action_dim * 2
        actor_mean = nn.Dense(action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        pi = self.dist(actor_mean)

        critic = nn.Dense(self.config.NUM_UNITS, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """A transition used in batch updates."""

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config: PPO_Params, logger: DummyLogger):
    """Create the training function."""
    config.NUM_EVALS = (config.episodes * config.update_steps * config.rollout_steps) // config.eval_every

    rnn_cls = globals()[config.MODEL]

    env, env_info, eval_env = make_env(config.env_params, make_eval=True)

    config.discrete = env_info["discrete"]

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.env_params.batch_size * config.UPDATE_EPOCHS)) / config.episodes * config.update_steps
        )
        return config.LR * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_size, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config.env_params.batch_size, env.observation_size)),
            jnp.zeros((1, config.env_params.batch_size)),
        )
        init_hstate = rnn_cls.initialize_carry(config.env_params.batch_size, config.NUM_UNITS)
        network_params = network.init(_rng, init_hstate, init_x)
        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip),
                optax.adam(config.LR, eps=1e-5),
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
        init_hstate = rnn_cls.initialize_carry(config.env_params.batch_size, config.NUM_UNITS)

        def eval_model(train_state, config: PPO_Params, seed=0):
            """Evaluate model."""
            rng = jax.random.PRNGKey(seed)
            rng, rng_init = jax.random.split(rng)

            env_state = eval_env.reset(rng_init)
            runner_state = (
                train_state,
                env_state,
                env_state.obs.reshape((1, -1)),
                jnp.zeros(1, dtype=bool),
                rnn_cls.initialize_carry(1, config.NUM_UNITS),
                _rng,
            )
            # COLLECT TRAJECTORIES

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = ActorCriticRNN(eval_env.action_size, config=config).apply(
                    train_state.params, hstate, ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                env_state = eval_env.step(env_state, action[0])
                transition = Transition(
                    last_done, action, value, env_state.reward.squeeze(), log_prob, last_obs, env_state.info
                )
                runner_state = (
                    train_state,
                    env_state,
                    env_state.obs.reshape((1, -1)),
                    env_state.done[None],
                    hstate,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.eval_steps)

            return traj_batch.reward.sum() / jnp.max(jnp.array([jnp.sum(traj_batch.done), 1]))

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis], last_done[np.newaxis])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                env_state = env.step(env_state, action)
                transition = Transition(
                    last_done, action, value, env_state.reward.squeeze(), log_prob, last_obs, env_state.info
                )
                runner_state = (train_state, env_state, env_state.obs, env_state.done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config.rollout_steps)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.gamma * next_value * (1 - next_done) - value
                    gae = delta + config.gamma * config.GAE_LAMBDA * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        init_hstate = jax.tree.map(lambda a: a[0], init_hstate)  # TBH
                        # RERUN NETWORK
                        _, pi, value = network.apply(params, init_hstate, (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.CLIP_EPS, config.CLIP_EPS
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config.VF_COEF * value_loss - config.ENT_COEF * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.env_params.batch_size)
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config.env_params.batch_size, -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = jax.tree.map(lambda a: a[None], initial_hstate)  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.UPDATE_EPOCHS)
            train_state = update_state[0]
            metric = {"rewards": traj_batch.reward, "dones": traj_batch.done, "step": train_state.step}
            rng = update_state[-1]

            # def callback(info):
            #     # FIXME: May become NaN!
            #     return_values = jnp.sum(info["rewards"]) / jnp.sum(info["dones"])
            #     # Logging -------------------------------------------------------------
            #     metrics = {
            #         "steps": info["step"] * config.rollout_steps * config.env_params.batch_size,
            #         "mean_reward": return_values,
            #     }
            #     # W & B logging
            #     logger.log(metrics, step=info["step"])

            # jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv.reshape((config.env_params.batch_size, -1)),
            jnp.zeros((config.env_params.batch_size), dtype=bool),
            init_hstate,
            _rng,
        )

        def log_eval_callback(metric, steps_since_best, timestep):
            loggables = {"eval/rewards": metric}
            if metric > float(logger["best_eval_reward"]):
                steps_since_best = jnp.zeros(())
                logger["best_eval_reward"] = float(metric)
                loggables["best_eval_reward"] = metric
            else:
                steps_since_best = steps_since_best.copy() + 1
            logger.log(loggables, step=timestep)
            print("Eval reward:", metric)
            print(f"Global step: {timestep}, eval reward: {metric:.2f}, best: {logger['best_eval_reward']:.2f}")
            return steps_since_best.astype(np.int32)

        steps_since_best = jnp.zeros((), dtype=np.int32)

        for _ in range(config.NUM_EVALS):
            runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config.update_steps)

            eval_reward = eval_model(runner_state[0], config)

            steps_since_best = jax.pure_callback(
                log_eval_callback,
                steps_since_best.copy(),
                eval_reward,
                steps_since_best.copy(),
                runner_state[0].step * config.rollout_steps * config.env_params.batch_size,
            )
            # Early stopping
            if config.patience and steps_since_best >= config.patience:
                print(f"Early stopping patience {config.patience}")
                break

        return {"runner_state": runner_state, "metric": metric}

    return train


def train_and_eval(config: PPO_Params, logger=DummyLogger()):
    """Run training."""
    rng = jax.random.PRNGKey(config.seed)
    logger["best_eval_reward"] = -np.inf
    try:
        make_train(config, logger)(rng)
        eval_reward = logger["best_eval_reward"]
    except Exception as e:
        raise e
    finally:
        logger.finalize()
    return eval_reward


if __name__ == "__main__":
    cmd_line_params = {}
    if len(sys.argv) > 1:
        cmd_line_params = json.loads(sys.argv[1])
    params = PPO_Params(**cmd_line_params)
    best_reward = with_logger(train_and_eval, params, logger_name="aim", project_name="PPO")
    print(f"Best eval reward: {best_reward:.2f}")
