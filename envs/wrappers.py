"""Wrappers for gym environments."""

import importlib
import os
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import gymnasium as gym
import gymnax
import jax
import numpy as np
from brax.envs.base import State
from jax import numpy as jnp
from jax import random as jrandom

from jax_rl_util.envs.env_util import make_obs_mask


def is_discrete(env: gym.Env):
    """Check if env has discrete Action Space."""
    return isinstance(env.action_space, gym.spaces.Discrete)


class Wrapper:
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: gym.Env):  # noqa
        self.env: gym.Env = env

    def __getattr__(self, name):  # noqa
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def discrete(self) -> bool:
        """Get action size."""
        return is_discrete(self)


class GymBraxWrapper(Wrapper):
    """Wrap Gym envs for use with Brax Wrappers."""

    def __init__(self, env, params=None):
        """Set Env params at initialization."""
        self.env = env
        self.params = params

    @property
    def unwrapped(self):
        """Unwrapped is self to stop recursion."""
        return self

    def reset(self, rng: jnp.ndarray) -> State:
        """Make brax state from gym reset output."""
        reset_key, step_key = jrandom.split(rng)
        obs, env_state = self.env.reset(reset_key)
        state = State(env_state, obs, jnp.zeros(1), False)
        state.info["rng"] = step_key
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Make gymnax step and wrap in brax state."""
        obs, gymnax_state, reward, done, _ = self.env.step(state.pipeline_state, action, self.params)
        # for k, v in state_gymnax[4].items():
        #     state.info[k] = v
        reward = jnp.array(reward, dtype=jnp.float32)
        if len(reward.shape) == 0:
            reward = jnp.expand_dims(reward, axis=0)
        return state.replace(pipeline_state=gymnax_state, obs=obs, reward=reward, done=done)

    @property
    def action_size(self) -> int:
        """Get action size."""
        act_space = self.action_space
        return act_space.n if self.discrete else act_space.shape[-1]

    @property
    def observation_size(self) -> int:
        """Only works for default_params envs."""
        return np.prod(self.observation_space.shape)


class GymnaxBraxWrapper(GymBraxWrapper):
    """Wrap Gymnax envs for use with Brax Wrappers."""

    def __init__(self, env, params: dict | None = None):
        """Set Env params at initialization."""
        self.env = env
        env_module = importlib.import_module(env.__class__.__module__)
        if params is None:
            self.params = env.default_params
        else:
            self.params = env_module.EnvParams(**params)

    def reset(self, rng: jnp.ndarray) -> State:
        """Make brax state from gym reset output and insert env params."""
        reset_key, step_key = jrandom.split(rng)
        obs, env_state = self.env.reset(reset_key, self.params)
        state = State(env_state, obs, jnp.zeros(1), False)
        state.info["rng"] = step_key
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Make gymnax step and wrap in brax state."""
        step_key, state.info["rng"] = jrandom.split(state.info["rng"])
        obs, gymnax_state, reward, done, _ = self.env.step_env(step_key, state.pipeline_state, action, self.params)
        # for k, v in state_gymnax[4].items():
        #     state.info[k] = v
        reward = jnp.array(reward, dtype=jnp.float32)
        if len(reward.shape) == 0:
            reward = jnp.expand_dims(reward, axis=0)
        return state.replace(pipeline_state=gymnax_state, obs=obs, reward=reward, done=done)

    @property
    def action_space(self, params=None) -> int:
        """Get action size."""
        params = self.env.default_params if params is None else params
        return self.env.action_space()

    @property
    def discrete(self) -> int:
        """Get action size."""
        return isinstance(self.action_space, gymnax.environments.spaces.Discrete)

    @property
    def observation_space(self, params=None) -> int:
        """Only works for default_params envs."""
        params = self.env.default_params if params is None else params
        return self.env.observation_space(params)


# class RenderWrapper(GymnaxToGymWrapper):
#     def __init__(self, env, params=None, seed: int | None = None):
#         super().__init__(env, params, seed)
#         self.vis = Visualizer(env, params, state_seq=None, reward_seq=None)

#     def render(self, mode="human"):
#         return super().render(mode)


class PopJymBraxWrapper(GymnaxBraxWrapper):
    """Wrap Gymnax envs for use with Brax Wrappers."""

    def __init__(self, env, params: dict | None = None):
        """Set Env params at initialization."""
        self.env = env
        env_module = importlib.import_module(env.__class__.__module__)
        if params is None:
            self.params = env.default_params
        else:
            self.params = env_module.EnvParams(**params)
            if hasattr(env_module, "MetaEnvParams"):
                self.params = env_module.MetaEnvParams(env_params=self.params)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end. Additionally allows action repetition."""

    def __init__(self, env, episode_length: int, action_repeat: int):
        """Initialize episode length and action repeat."""
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jnp.ndarray) -> State:
        """Set step count and truncation to zero."""
        state = self.env.reset(rng)
        state.info["steps"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.zeros(rng.shape[:-1], dtype=jnp.int32)
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Check if max episode length was reached and optionally do action repetition."""

        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jnp.int32(jnp.where(steps >= episode_length, 1 - state.done, zero))
        state.info["steps"] = steps
        return state.replace(done=done)


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env, batch_size: int = None):
        """Set batch size."""
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jnp.ndarray) -> State:
        """Split rng and vmap over reset."""
        if self.batch_size is not None and len(rng.shape) == 1:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Vmap over step."""
        return jax.vmap(self.env.step)(state, action)


class EfficientAutoResetWrapper(Wrapper):
    """Efficiently resets Brax envs that are done.

    Attention! The first state is remembered and used to reset the env.
    """

    def reset(self, rng: jnp.ndarray) -> State:
        """Remember first state."""
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Reset to remembered first state if done."""
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.jax.tree.map(where_done, state.info["first_pipeline_state"], state.pipeline_state)
        obs = where_done(state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class GymJaxWrapper(Wrapper):
    """Wrap Gym envs for use with Jax."""

    @property
    def unwrapped(self):
        """Unwrapped is self to stop recursion."""
        return self

    def reset(self, rng: jnp.ndarray):
        """Call gym reset as external callback."""
        result_shape_dtypes = (
            jax.ShapeDtypeStruct(self.env.observation_space.shape, dtype=self.env.observation_space.dtype),
            {},
        )
        return jax.experimental.io_callback(self.env.reset, result_shape_dtypes)

    def step(self, state, action: jnp.ndarray, key: jrandom.PRNGKey = None):
        """Make gymnax step and wrap in brax state."""
        result_shape_dtypes = (
            jax.ShapeDtypeStruct(self.env.observation_space.shape, dtype=self.env.observation_space.dtype),
            jax.ShapeDtypeStruct((), dtype=jnp.float32),
            jax.ShapeDtypeStruct((), dtype=jnp.bool),
        )

        def _step(action):
            obs, reward, done, truncated, info = self.env.step(action)
            # FIXME: Cannot pass back info with autoreset since shape changes
            return obs, jnp.array(reward, dtype=jnp.float32), jnp.array(done or truncated)

        obs, reward, done = jax.experimental.io_callback(_step, result_shape_dtypes, action)
        return obs, state, reward, done

    @property
    def action_size(self) -> int:
        """Get action size."""
        act_space = self.action_space
        return act_space.n if self.discrete else act_space.shape[-1]

    @property
    def observation_size(self) -> int:
        return self.observation_space.shape


class RandomizedAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done.

    Force resample every step. Inefficient
    """

    def reset(self, rng: jnp.ndarray) -> State:
        """Make brax state from gym reset output."""
        reset_key, step_key = jrandom.split(rng)
        state = self.env.reset(reset_key)
        state.info["rng"] = step_key
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Resample new initial state for all parallel envs."""
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        rng, _rng = jrandom.split(state.info["rng"])
        done = state.done

        def _reset():
            reset_state = self.env.reset(_rng)
            reset_state.info["rng"] = rng
            return reset_state

        state = jax.lax.cond(state.done, _reset, lambda: state)
        return state.replace(done=done)


class RandomizedAutoResetWrapperNaive(Wrapper):
    """Automatically resets Brax envs that are done.

    Force resample every step. Inefficient
    """

    def reset(self, rng: jnp.ndarray) -> State:
        """Make brax state from gym reset output."""
        reset_key, step_key = jrandom.split(rng)
        state = self.env.reset(reset_key)
        if hasattr(self, "batch_size"):
            step_key = jax.random.split(step_key, self.batch_size)
        state.info["rng"] = step_key
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Resample new initial state for all parallel envs."""
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        reset_state = self.env.reset(state.info["rng"])

        def where_done(new, old):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [new.shape[0]] + [1] * (len(new.shape) - 1))
            return jnp.where(done, new, old)

        pipeline_state = jax.jax.tree.map(where_done, reset_state.pipeline_state, state.pipeline_state)
        obs = where_done(reset_state.obs, state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class FlatPOWrapper(Wrapper):
    """Flattens and Masks Observations in order to create an POMDP."""

    def __init__(self, env: gym.Env, obs_mask: Iterable[int] | str):
        """Set obs_mask."""
        super().__init__(env)
        self.obs_mask = make_obs_mask(np.prod(env.observation_space.shape), obs_mask)

    def reset(self, rng: jnp.ndarray):
        """Mask Observation."""
        obs, env_state = self.env.reset(rng)
        return obs.reshape(-1)[..., self.obs_mask], env_state

    def step(self, rng, state, action: jnp.ndarray):
        """Mask Observation."""
        obs, env_state, reward, done = self.env.step(rng, state, action)
        return obs.reshape(-1)[..., self.obs_mask], env_state, reward, done

    @property
    def observation_size(self):
        """Get the size of the masked Observation."""
        return (len(self.obs_mask),)


class FlatObs_BraxWrapper(Wrapper):
    """Flattens Observations."""

    def reset(self, rng: jnp.ndarray) -> State:
        """Flatten Observations."""
        state = self.env.reset(rng)
        state.info["fo_obs"] = state.obs
        return state.replace(obs=state.obs.reshape((-1)))

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Flatten Observations."""
        state = state.replace(obs=state.info.get("fo_obs", state.obs))
        state = self.env.step(state, action)
        state.info["fo_obs"] = state.obs
        return state.replace(obs=state.obs.reshape((-1)))


class POBraxWrapper(Wrapper):
    """Masks Observations in order to create an POMDP."""

    def __init__(self, env, obs_mask: Iterable[int]):
        """Set obs_mask."""
        super().__init__(env)
        self.obs_mask = make_obs_mask(np.prod(env.observation_size), obs_mask)

    def reset(self, rng: jnp.ndarray) -> State:
        """Mask Observation."""
        state = self.env.reset(rng)
        state.info["full_obs"] = state.obs
        return state.replace(obs=state.obs[..., self.obs_mask])

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Mask Observation."""
        state = self.env.step(state, action)
        return state.replace(obs=state.obs[..., self.obs_mask])

    @property
    def observation_size(self):
        """Get the size of the masked Observation."""
        return len(self.obs_mask)

    @property
    def action_size(self):
        """Get the size of the masked Observation."""
        return self.env.action_size

    @property
    def backend(self):
        """Get the size of the masked Observation."""
        return self.env.backend


@dataclass
class LogEnvState:
    """Log state with episode returns and lengths."""

    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper:
    """Log the episode returns and lengths."""

    def __init__(self, env):  # noqa
        self._env = env

    def __getattr__(self, name):
        """Provide proxy access to regular attributes of wrapped object."""
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        """Also reset returns and lengths."""
        obs, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: State, action):
        """Take a step and log the returns and lengths."""
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class RGBtoGrayWrapper(Wrapper):
    """Converts RGB to Grayscale."""

    def _convert(self, obs: jnp.ndarray) -> jnp.ndarray:
        # Scale to [0, 1]
        obs = obs / 255.0
        # Convert to grayscale
        import dm_pix as pix

        return jax.vmap(pix.rgb_to_grayscale)(obs) - 0.5

    def reset(self, rng):
        """Convert RGB to Grayscale."""
        out = self.env.reset(rng)
        return (self._convert(out[0]), *out[1:])

    def step(self, state, action: jnp.ndarray, key: jrandom.PRNGKey = None):
        """Convert RGB to Grayscale."""
        out = self.env.step(state, action, key)
        return (self._convert(out[0]), *out[1:])

    @property
    def observation_space(self) -> int:
        return gym.Space(self.env.observation_space.shape[:-1] + (1,), dtype=float)


class ParamWrapper(GymJaxWrapper):
    """Wrap Gymnax envs for use with Brax Wrappers."""

    def __init__(self, env, params=None):
        """Set Env params at initialization."""
        self.env = env
        env_module = importlib.import_module(env.__class__.__module__)
        if params is None:
            self.params = env.default_params
        else:
            self.params = env_module.EnvParams(**params)

    def reset(self, rng: jnp.ndarray):
        """Make brax state from gym reset output and insert env params."""
        return self.env.reset(rng, self.params)

    def step(self, key, state, action: jnp.ndarray):
        """Make gymnax step and wrap in brax state."""
        return self.env.step(key, state, action, self.params)


class JitWrapper(Wrapper):
    """Jit wrapped jax env."""

    def reset(self, *inputs):
        return jax.jit(self.env.reset)(*inputs)

    def step(self, *inputs):
        return jax.jit(self.env.step)(*inputs)


class SaveToFileWrapper(Wrapper):
    """This wrapper saves observations, actions and rewards to file(s)."""

    def __init__(
        self,
        env: gym.Env,
        output_folder: str,
        min_steps: int = 2,
        start_filenum: int = 0,
    ):
        """Wrapper records arrays of rollouts.

        For now will save one episode per file.

        Args:
            env: The environment that will be wrapped
            output_folder (str): The folder where the rollouts will be stored

        """
        gym.Wrapper.__init__(self, env)

        self.min_steps = min_steps

        self.output_folder = os.path.abspath(output_folder)
        # Create output folder if needed
        if os.path.isdir(self.output_folder):
            print(f"Overwriting existing files in {self.output_folder}")
        os.makedirs(self.output_folder, exist_ok=True)

        try:
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.is_vector_env = False

        self.obs_buffer = []
        self.act_buffer = []
        self.rew_buffer = [0]
        self.filenum = start_filenum

    def _save_rollout(self):
        """Save the current episode to file."""
        if len(self.obs_buffer) >= self.min_steps:
            obs = np.array(self.obs_buffer)
            act = np.array(self.act_buffer)
            rew = np.array(self.rew_buffer)
            filename = os.path.join(self.output_folder, f"episode-{self.filenum}.npz")
            np.savez(filename, obs=obs, act=act, rew=rew)
            self.filenum += 1

        # In any case, wipe the buffers
        self.obs_buffer = []
        self.act_buffer = []
        self.rew_buffer = [0]

    def reset(self, *args, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        self._save_rollout()
        state = self.env.reset(*args, **kwargs)
        self.obs_buffer.append(state.obs)
        return state

    def step(self, state, action: jnp.ndarray):
        """Steps through the environment using action, recording actions, observations and rewards"""
        env_state = self.env.step(state, action)

        self.act_buffer.append(action)

        if env_state.done:
            self._save_rollout()

        # Usually with Autoreset, the returned obs is the start of the next episode
        self.obs_buffer.append(env_state.obs)
        self.rew_buffer.append(env_state.reward)
        return env_state

    def close(self):
        """Closes the wrapper then the video recorder."""
        self._save_rollout()
        super().close()
