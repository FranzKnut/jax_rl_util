"""Classic cart-pole system implemented by Rich Sutton et al.

adjusted for gymnax by jlemmel
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from functools import partial
import gymnasium
import gymnax
import jax
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from jax import numpy as jnp
from jax import random as jrandom
from flax import struct


@struct.dataclass
class EnvParams(gymnax.environments.EnvParams):
    force_mag: float = 5.0
    tau: float = 0.02  # seconds between state updates
    length: float = 0.5  # length to center of mass
    masscart: float = 1.0
    masspole: float = 0.1
    x_threshold: float = 2.4
    theta_threshold_radians: float = 12 * 2 * jnp.pi / 360
    pole_friction: float = 0.01  # kg m² / s²
    momentum_inertia: float = 1.0e-2  # kg m²


class ContinuousCartPoleEnv(CartPoleEnv, gymnax.environments.environment.Environment):
    """A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.

    The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing.
        This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    @property
    def default_params(self):
        if self.task.endswith("dv"):
            _p = EnvParams(
                force_mag=4.0,
                tau=0.01,
                length=0.41,
                masscart=0.46,
                masspole=0.08,
                x_threshold=0.4,
                pole_friction=2.1e-3,
                momentum_inertia=1.05e-2,
            )
        else:
            _p = EnvParams()
        if self.task.startswith("damping"):
            _p = _p._replace(tau=0.02, pole_friction=0.1, momentum_inertia=0.1)
        return _p

    def __init__(self, seed=0, task="balancing-dv"):
        """Initialize environment.

        Args:
            seed (int, optional): Random seed. Defaults to 0.
            task (str, optional): Task to solve. Defaults to 'balancing'.
                                  Options: 'balancing', 'balancing-dv', 'damping', 'damping-dv'
                                  suffix 'dv' uses env parameters from a real-world beckhoff system.
        """
        self.key = jrandom.PRNGKey(seed)
        super().__init__(render_mode="rgb_array")
        del self.observation_space
        del self.action_space

        self.start_theta = 0.0
        self.task = task

    def observation_space(self, params=None):
        if params is None:
            params = self.default_params

        high = jnp.array(
            [
                params.x_threshold * 2,
                jnp.inf,
                params.theta_threshold_radians * 2,
                jnp.inf,
            ],
            dtype=jnp.float32,
        )

        return gymnax.environments.spaces.Box(-high, high, shape=(4,), dtype=jnp.float32)

    def action_space(self, params=None):
        return gymnax.environments.spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)

    @partial(jax.jit, static_argnums=0)
    def _step(self, state, action, params=None):
        if params is None:
            params = self.default_params
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        xacc = params.force_mag * action.squeeze()
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        # temp = (force + params.polemass_length * theta_dot * theta_dot * sintheta) / params.total_mass
        thetaacc = (
            params.masspole
            * params.length
            * (self.gravity * sintheta - xacc * costheta)
            - params.pole_friction * theta_dot
        ) / params.momentum_inertia
        # xacc = temp - params.polemass_length * thetaacc * costheta / params.total_mass
        if self.kinematics_integrator == "euler":
            x = x + params.tau * x_dot
            x_dot = x_dot + params.tau * xacc
            theta = theta + params.tau * theta_dot
            theta_dot = theta_dot + params.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + params.tau * xacc
            x = x + params.tau * x_dot
            theta_dot = theta_dot + params.tau * thetaacc
            theta = theta + params.tau * theta_dot
        state = jnp.array([x, x_dot, theta, theta_dot])
        done = (x < -params.x_threshold) | (x > params.x_threshold)
        if self.task.startswith("balancing"):
            done = done | (jnp.abs(theta) > self.theta_threshold_radians)

        reward = self.get_reward(state, action)
        return state, state, reward, done, jnp.zeros((), dtype=bool)

    def get_reward(self, state, action):
        return 1.0

    @staticmethod
    def clip_theta(state):
        if state[2] > np.pi:
            state[2] = -2 * np.pi + state[2]
        elif state[2] <= -np.pi:
            state[2] = 2 * np.pi + state[2]
        return state

    def step_env(self, key, state, action=None, params=None):
        if action is None:
            # For compatibiliy with regular gym
            action = state
            state = self.state
        # action = np.clip(action,-1,1)[0]
        output = self._step(state, action, params=params)
        self.elapsed_time = self.elapsed_time + self.tau
        self.state = output[0]
        return output

    def _reset(self, key, params=None):
        if params is None:
            params = self.default_params
        if self.task.startswith("balancing"):
            bounds = jnp.array(
                [
                    params.x_threshold / 2,
                    0.05,
                    params.theta_threshold_radians / 2,
                    0.05,
                ],
                dtype=jnp.float32,
            )
        else:
            bounds = jnp.array([0.05, 0.05, jnp.pi / 2, 0.05])

        initial_state = jrandom.uniform(key, minval=-bounds, maxval=bounds, shape=(4,))
        initial_state = initial_state.at[2].set(initial_state[2] + self.start_theta)
        return initial_state

    def reset(self, seed=None, params=None):
        if seed is None:
            self.key, seed = jrandom.split(self.key)

        self.elapsed_time = 0.0
        state = self._reset(seed, params=params)
        self.state = state

        return state, state


class CartPoleSwingUp(ContinuousCartPoleEnv):
    def __init__(
        self,
        offcenter_penalty_factor=1e-3,
        theta_dot_penalty_factor=1e-3,
        switch_x_dir_penalty_factor=1e-2,
    ):
        super().__init__(task="swingup")
        self.start_theta = np.pi
        self.theta_threshold_radians = np.pi / 2
        self.offcenter_penalty_factor = offcenter_penalty_factor
        self.theta_dot_penalty_factor = theta_dot_penalty_factor
        self.switch_x_dir_penalty_factor = switch_x_dir_penalty_factor

    def get_reward(self, state, action):
        x, _, theta, theta_dot = state
        is_above = jnp.abs(theta) < (jnp.pi / 2)
        reward = jnp.array(
            [
                1 + jnp.cos(theta),
                -self.theta_dot_penalty_factor
                * jnp.abs(theta_dot)
                * jnp.cos(theta)
                * is_above,
                -self.offcenter_penalty_factor * jnp.abs(x) * jnp.cos(theta) * is_above,
                -self.switch_x_dir_penalty_factor
                * ((x * action.squeeze()) < 0)
                * jnp.sin(theta)
                * (1 - is_above),
            ]
        ).sum()
        return reward

    def is_terminated(self, state):
        return state[0] < -self.x_threshold or state[0] > self.x_threshold


class CartPoleDampening(ContinuousCartPoleEnv):
    """CartPoleDecoupled for dampening task"""

    def __init__(self):
        super().__init__(task="damping-dv")
        self.name = "CartpoleContinuousJaxSwingUp-v0"
        self.theta_threshold_radians = 0.5
        self.start_x_bound = 0.3

    def get_reward(self, state, action):
        x, _, theta, theta_dot = state
        reward = jnp.sum(
            [
                1 - jnp.cos(theta),
                -0.01 * jnp.abs(theta_dot),
                -0.5 * jnp.abs(x),
            ]
        )
        return reward

    def is_terminated(self, state):
        return state[0] < -self.x_threshold or state[0] > self.x_threshold


gymnasium.register(
    id="CartpoleContinuousJax-v0",
    entry_point=ContinuousCartPoleEnv,
    order_enforce=False,
)
gymnasium.register(
    id="CartpoleContinuousJaxSwingUp-v0",
    entry_point=CartPoleSwingUp,
    order_enforce=False,
)
