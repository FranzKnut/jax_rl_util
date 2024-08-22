"""
Classic cart-pole system implemented by Rich Sutton et al.

adjusted for jax by jlemmel
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

from functools import partial
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import spaces
from jax import numpy as jnp
from jax import random as jrandom
import jax
import numpy as np

gym.envs.register(
    id='CartpoleContinuousJax-v0',
    entry_point='experiments.reinforcement.envs.continuous_cartpole:ContinuousCartPoleEnv',
    order_enforce=False
)


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

    def __init__(self, seed=0, task='balancing'):
        """Initialize environment.
        
        Args:
            seed (int, optional): Random seed. Defaults to 0.
            task (str, optional): Task to solve. Defaults to 'balancing'.
                                  Options: 'balancing', 'balancing-dv', 'damping', 'damping-dv'
                                  suffix 'dv' uses env parameters from a real-world beckhoff system.
        """
        self.key = jrandom.PRNGKey(seed)
        self.name = "CartPole-v1"
        super().__init__(render_mode='rgb_array')
        if task.endswith('dv'):
            self.force_mag = 3.0
            self.tau = 0.01
            self.length = 0.2
            self.masscart = 0.46
            self.masspole = 0.08
            self.x_threshold = 0.4
        if task.startswith('damping'):
            self.theta_threshold_radians = 30 * jnp.pi / 360
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=jnp.float32)
        self.task = task

    @partial(jax.jit, static_argnums=0)
    def _step(self, state, action):
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        force = self.force_mag * action.squeeze()
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
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
        if self.task.startswith('balancing'):
            done = done | (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)

        reward = (1-done)
        if self.task.startswith('damping'):
            theta_wrapped = (theta + np.pi) % (2 * np.pi) - np.pi
            theta_symmetric = abs(theta_wrapped)
            distance_to_target = abs(theta_symmetric - np.pi)
            distance_to_center = abs(x)
            reward += distance_to_target * (done - 1) / 10
            reward += distance_to_center * (done - 1) / 10
            reward += abs(theta_dot) * (done - 1) / 10
            reward += abs(x_dot) * (done - 1) / 10

        return state, state, reward, done, False

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
        if self.task.startswith('balancing'):
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
            bounds = jnp.array([0.05, 0.05, np.pi/2, 0.05])

        return jrandom.uniform(key, minval=-bounds, maxval=bounds, shape=(4,))

    def reset(self, key=None):
        if key is None:
            self.key, key = jrandom.split(self.key)

        self.elapsed_time = 0.0
        state = self._reset(key)
        self.state = state

        return state, state
