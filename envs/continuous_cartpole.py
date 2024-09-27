"""
Classic cart-pole system implemented by Rich Sutton et al.

adjusted for jax by jlemmel
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from functools import partial
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
from jax import numpy as jnp
from jax import random as jrandom
import jax
import numpy as np


class ContinuousCartPoleEnv(CartPoleEnv):
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

    def __init__(self, seed=0, task="balancing-dv", action_mode="acc"):
        """Initialize environment.

        Args:
            seed (int, optional): Random seed. Defaults to 0.
            task (str, optional): Task to solve. Defaults to 'balancing'.
                                  Options: 'balancing', 'balancing-dv', 'damping', 'damping-dv'
                                  suffix 'dv' uses env parameters from a real-world beckhoff system.
        """
        self.key = jrandom.PRNGKey(seed)
        self.name = "CartpoleContinuousJax-v0"
        super().__init__(render_mode="rgb_array")

        self.start_theta = 0
        if task.endswith("dv"):
            self.force_mag = 4.0
            self.tau = 0.01
            self.length = 0.41
            self.masscart = 0.46
            self.masspole = 0.08
            self.x_threshold = 0.4
            self.pole_friction = 2.1e-3  # kg m² / s²
            self.momentum_inertia = 1.05e-2  # kg m^2

        if task.startswith("damping"):
            self.theta_threshold_radians = 30 * jnp.pi / 360
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)
        self.task = task

    @partial(jax.jit, static_argnums=0)
    def _step(self, state, action):
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        xacc = self.force_mag * action.squeeze()
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        # temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (
            self.masspole * self.length * (self.gravity * sintheta - xacc * costheta) - self.pole_friction * theta_dot
        ) / self.momentum_inertia
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        state = jnp.array([x, x_dot, theta, theta_dot])
        done = (x < -self.x_threshold) | (x > self.x_threshold)
        if self.task.startswith("balancing"):
            done = done | (jnp.abs(theta) > self.theta_threshold_radians)

        reward = self.get_reward(state, action)

        return state, reward, done, False, {}

    def get_reward(self, state, action):
        return 1

    @staticmethod
    def clip_theta(state):
        if state[2] > np.pi:
            state[2] = -2 * np.pi + state[2]
        elif state[2] <= -np.pi:
            state[2] = 2 * np.pi + state[2]
        return state

    def step(self, state, action=None, params=None):
        if action is None:
            # For compatibiliy with regular gym
            action = state
            state = self.state
        # action = np.clip(action,-1,1)[0]
        output = self._step(state, action)
        self.elapsed_time += self.tau
        self.state = output[0]
        return output

    def _reset(self, key):
        if self.task.startswith("balancing"):
            bounds = jnp.array(
                [
                    self.x_threshold / 2,
                    0.05,
                    self.theta_threshold_radians / 2,
                    0.05,
                ],
                dtype=np.float32,
            )
        else:
            bounds = jnp.array([0.05, 0.05, np.pi / 2, 0.05])

        initial_state = jrandom.uniform(key, minval=-bounds, maxval=bounds, shape=(4,))
        initial_state = initial_state.at[2].set(initial_state[2] + self.start_theta)
        return initial_state

    def reset(self, seed=None):
        if seed is None:
            self.key, seed = jrandom.split(self.key)

        self.elapsed_time = 0.0
        state = self._reset(seed)
        self.state = state

        return state, {"state": state}


class CartPoleSwingUp(ContinuousCartPoleEnv):
    def __init__(
        self,
        offcenter_penalty_factor=1e-3,
        theta_dot_penalty_factor=1e-3,
        switch_x_dir_penalty_factor=1e-2,
    ):
        super().__init__(task="swingup-dv")
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
                -self.theta_dot_penalty_factor * jnp.abs(theta_dot) * jnp.cos(theta) * is_above,
                -self.offcenter_penalty_factor * jnp.abs(x) * jnp.cos(theta) * is_above,
                -self.switch_x_dir_penalty_factor * ((x * action.squeeze()) < 0) * jnp.sin(theta) * (1 - is_above),
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
        reward = np.sum(
            [
                1 - np.cos(theta),
                -0.01 * np.abs(theta_dot),
                -0.5 * np.abs(x),
            ]
        )
        return reward

    def is_terminated(self, state):
        return state[0] < -self.x_threshold or state[0] > self.x_threshold
