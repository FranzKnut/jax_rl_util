#!/usr/bin/env python3
"""A gym environment for drone simulations."""

import argparse
import csv
import pathlib
from collections import OrderedDict
from typing import Tuple

import gymnax
import gymnax.environments.spaces
import jax.numpy as jnp
import jax.random as jrandom
from flax import struct
from gymnax.environments.environment import Environment as GymnaxEnv


@struct.dataclass
class EnvParams:
    """Class representing the parameters for the DroneGym environment.

    Attributes
    ----------
        output (str): Output file name.
        frequency (int): Frequency of the environment.
        dims (int): Number of dimensions.
        ndrones (int): Number of drones.
        initial_velocity_stddev (float): Standard deviation of initial velocity for the ego drone.
        ego_change_velocity_stddev (float): Standard deviation of velocity change for the ego drone.
        velocity_stddev (float): Standard deviation of velocity for other drones.
        change_velocity_stddev (float): Standard deviation of velocity change for other drones.
        noise_stddev (float): Standard deviation of distance measurement noise.
        noise_color (int): Color of the noise.
        initial_pos_stddev (float): Standard deviation of initial position distribution for other drones.
        iir_filter (float): IIR filter value.
        noise_iir_value (float): IIR value for noise.
        goto_stddev (float): Standard deviation for goto action.
        n_drones (int): Number of drones.
        n_dim (int): Number of dimensions.
        starting_post_ego (Tuple[float, float, float]): Starting position of the ego drone.
        plot_range (int): Range for plotting.
        steps (int): Number of steps in the environment.
    """

    frequency: int = 30
    max_steps: int = 100
    action_mode: int = 0  # 0 = acc, 1 = vel

    # velocity parameters for ego drone
    action_scale: float = 1

    # initial_velocity_stddev: float = 0.1
    ego_change_velocity_stddev: float = 0.005

    # velocity parameters for other drones
    change_velocity_stddev: float = 0

    # distance measurement noise
    noise_stddev: float = 0.15
    noise_color: int = 0

    # initial position distribution for other drones
    initial_pos_stddev: float = 2.0
    initial_vel_stddev: float = 0.5
    iir_filter: float = 0.2
    noise_iir_value: float = 0
    goto_stddev: float = 25.0

    n_drones: int = 1
    n_dim: int = 2
    starting_pos_ego: Tuple[float, float, float] = (5.0, 0.0, 0.0)
    starting_pos_goal: Tuple[float, float, float] = (-5.0, 0.0, 0.0)
    plot_range: int = 10

    # Difficulties
    include_pos_in_obs: bool = False
    obstacle: bool = True
    obstacle_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_size: float = 0.5
    failed_penalty: float = -100


# class EnvState


class DroneGym(GymnaxEnv):
    """A gym environment for drone simulations.

    Parameters
    ----------
    - params: A `DroneGymParams` object containing the parameters for the environment.

    Methods
    -------
    - __init__(self, params: DroneGymParams): Initializes the DroneGym object.
    - initial_state(self, rng_key): Generates the initial state of the environment.
    - apply_noise(self, distance, rng_key): Applies noise to the given distance.
    - step(self, rng_key, state, action=None): Performs a step in the environment.
    - _sample_observation(self, pos, vel, rng_key): Samples an observation from the environment.
    - reset(self, rng_key): Resets the environment to its initial state.
    """

    @property
    def default_params(self):
        """Default params."""
        return EnvParams()

    def __init__(self, params: EnvParams = EnvParams()):
        """Initialize the DroneGym object.

        Arguments:
        ---------
        params: A `DroneGymParams` object containing the parameters for the environment.
        """
        # initialize empty arrays
        self._step = 0
        assert params.n_dim in [2, 3], "Only 2D and 3D is supported"
        self.params = params
        self.dt = 1 / params.frequency
        # initialize ego and other drones
        self.starting_pos_ego = jnp.array(params.starting_pos_ego)[: params.n_dim]
        self.starting_pos_goal = jnp.array(params.starting_pos_goal)[: params.n_dim]
        self.obstacle_pos = jnp.array(params.obstacle_pos)[: params.n_dim]

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        lim = jnp.ones(params.n_dim)
        return gymnax.environments.spaces.Box(-lim, lim, shape=(params.n_dim,))

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return gymnax.environments.spaces.Box(
            -params.plot_range, params.plot_range, shape=2 + (params.n_dim if params.include_pos_in_obs else 0)
        )

    def state_space(self, params):
        """State space of the environment."""
        raise NotImplementedError

    def initial_state(self, rng_key):
        """Generate the initial state of the environment.

        Arguments:
        ---------
        rng_key: The random number generator key.

        Returns:
        -------
        - An `OrderedDict` representing the initial state of the environment.
        """
        k_pos, k_step = jrandom.split(rng_key)
        initial_pos = jnp.concatenate(
            [
                self.starting_pos_ego[None]
                + self.params.initial_pos_stddev * jrandom.normal(k_pos, [1, self.params.n_dim]),
                self.starting_pos_goal[None],
            ],
            axis=0,
        )
        initial_vel = jnp.concatenate(
            [
                self.params.initial_vel_stddev * jrandom.normal(k_pos, [self.params.n_drones, self.params.n_dim]),
                jnp.zeros_like(self.starting_pos_goal[None]),
            ],
            axis=0,
        )
        difference = initial_pos[0] - initial_pos[1:]
        true_distance = jnp.linalg.norm(difference, axis=-1)
        return OrderedDict(
            {
                "step": 0,
                "pos": initial_pos,
                "vel": initial_vel,
                "goto": jnp.zeros([self.params.n_drones, self.params.n_dim]),
                "reached_goal": False,
                "filtered_dist": true_distance,
                "rng": k_step,
            }
        )

    def apply_noise(self, distance, rng_key):
        """Apply noise to the given distance.

        Arguments:
        ---------
        distance: The distance to which noise is applied.
        rng_key: The random number generator key.

        Returns:
        -------
        The distance with noise applied.
        """
        if self.params.noise_color == 0:  # use white noise
            return distance + self.params.noise_stddev * jrandom.normal(rng_key)
        elif self.params.noise_color == 2:  # use pink noise A
            k1, k2 = jrandom.split(rng_key)
            rand_value = self.params.noise_stddev * jrandom.normal(k1)
            self.params.noise_iir_value = 0.99 * self.params.noise_iir_value + 0.01 * rand_value
            rand_value = self.params.noise_stddev * jrandom.normal(k2)
            return distance + (self.params.noise_iir_value + rand_value / 100) * 12  # add distance noise
        elif self.params.noise_color == 3:  # use pink noise B
            k1, k2 = jrandom.split(rng_key)
            rand_value = self.params.noise_stddev * jrandom.normal(k1)
            self.params.noise_iir_value = 0.995 * self.params.noise_iir_value + 0.005 * rand_value
            rand_value = self.params.noise_stddev * jrandom.normal(k2)
            return distance + (self.params.noise_iir_value + rand_value / 30) * 19.7  # add distance noise
        else:
            assert False, "Invalid noise option!"

    def step_env(self, key, state, action, params: EnvParams):
        """Perform a step in the environment.

        Arguments:
        ---------
        key: jax PRNG key.
        state: The current state of the environment.
        action: The action to be performed (optional).
        params: Current env params.

        Returns:
        -------
        - obs: The observation after the step.
        - state: The updated state of the environment.
        - reward: The reward obtained from the step.
        - done: A boolean indicating if the episode is done.
        - info: Additional information about the step.
        """
        # perform ego and other drone movement for next step
        step, pos, vel, goto, reached_goal, filtered_dist, key = state.values()
        ego_key, obs_key, drone_key, next_key = jrandom.split(key, 4)
        ego_vel = vel[0]

        # Random movement for other drones
        drones_acc = jrandom.normal(drone_key, [params.n_drones, params.n_dim]) * params.change_velocity_stddev
        drones_vel = vel[1:] + drones_acc

        # controlled movement for ego drone
        # controlled_movement_a
        #     vel = vel + jrandom.normal(rng_key, [params.n_dim]) * params.change_velocity_stddev

        #     lim_stddev = 15
        #     lim_factor = 500
        #     jnp.where(jnp.abs(pos) > jrandom.normal(0.0, lim_stddev), vel[0] -= vel[0] / lim_factor, vel[0])
        #     if (abs(params.position[0]) > jrandom.normal(0.0, lim_stddev)):
        #         vel[0] -= vel[0] / lim_factor
        #     if (abs(self.position[1]) > jrandom.normal(0.0, lim_stddev)):
        #         self.velocity[1] -= self.position[1] / lim_factor
        #     if (abs(self.position[2]) > jrandom.normal(0.0, lim_stddev)):
        #         self.velocity[2] -= self.position[2] / lim_factor

        # controlled_movement_b
        if params.action_mode == 0:
            ego_vel += action * params.action_scale
        elif params.action_mode == 1:
            ego_vel = action * params.action_scale
        else:
            raise ValueError("Unknown action_mode")
        # goto = ego_vel + jnp.where(step % 10 == 0, jrandom.normal(ego_key, [params.n_dim]) * params.goto_stddev, goto)
        # ego_vel = ego_vel * 0.99 + (goto - ego_pos) * 0.0004

        # Concatenate ego and drones velocities
        vel = jnp.concatenate([ego_vel[None], drones_vel], axis=0)
        pos += vel * self.dt

        # Sample observation
        obs, dists = self._sample_observation(pos, vel, obs_key)
        (
            goal_distance,
            noisy_goal_dist,
            obstacle_distance,  # true distance to drone 0
            noisy_obstacle_dist,  # noisy distance to drone 0
        ) = dists

        # filtered_dist = filtered_dist * (1.0 - params.iir_filter) + noisy_distance * (params.iir_filter)
        is_out_of_time = step >= params.max_steps

        # Reward when target is reached
        reward = 0.1 / jnp.max(jnp.array([1, noisy_goal_dist.squeeze()])) ** 2
        # reward = 0
        is_outside = jnp.any(jnp.abs(pos) > params.plot_range)
        is_at_target = (goal_distance <= 1).squeeze()
        reward = jnp.where(is_at_target & ~reached_goal, params.max_steps - step, reward)

        # If reached the target, rotate vector pointing to goal pos by 90 degrees
        # rotated_goal = jnp.array([-pos[1, 0], pos[1, 1]])
        # pos = pos.at[1].set(rotated_goal * is_at_target + pos[1] * (1 - is_at_target))

        done = is_outside | is_out_of_time
        failed = is_outside

        if params.obstacle:
            dist_to_obstacle = jnp.linalg.norm(pos[0] - jnp.array(params.obstacle_pos[: params.n_dim]))
            hit_obstacle = dist_to_obstacle <= params.obstacle_size
            done |= hit_obstacle
            failed |= hit_obstacle

        reward = jnp.where(failed, params.failed_penalty, reward)

        state = OrderedDict(
            {
                "step": step + 1,
                "pos": pos,
                "vel": vel,
                "goto": goto,
                "reached_goal": reached_goal | is_at_target,
                "filtered_dist": filtered_dist,
                "rng": next_key,
            }
        )
        return (
            obs,
            state,
            reward,
            done,
            {
                "data_ego_pos": state["pos"][0],
                "data_ego_vel": state["vel"][0],
                "data_drones_pos": state["pos"][1:],
                "data_true_distance_goal": goal_distance,
                "data_noisy_distance_goal": noisy_goal_dist,
                "data_true_distance_obst": obstacle_distance,
                "data_noisy_distance_obst": noisy_obstacle_dist,
            },
        )

    def _sample_observation(self, pos, vel, rng_key):
        """Sample an observation from the environment.

        Arguments:
        ---------
        pos: The positions of the drones.
        vel: The velocities of the drones.
        rng_key: The random number generator key.

        Returns:
        -------
        - An observation tuple containing the ego drone velocity, ego drone position, noisy distance to drone 0,
          true distance to drone 0, and other relevant information.
        """
        obstacle_distance = jnp.linalg.norm(pos[0] - self.obstacle_pos)
        noisy_obstacle_dist = self.apply_noise(obstacle_distance, rng_key)

        goal_distance = jnp.linalg.norm(pos[0] - pos[1:], axis=-1)
        noisy_goal_dist = self.apply_noise(goal_distance, rng_key)

        obs = jnp.concatenate(jnp.array([noisy_obstacle_dist[None], noisy_goal_dist]))
        if self.params.include_pos_in_obs:
            obs = jnp.append(pos[0], obs)

        return (
            obs,
            (
                goal_distance,
                noisy_goal_dist,
                obstacle_distance,  # true distance to drone 0
                noisy_obstacle_dist,  # noisy distance to drone 0
                # ego.get_distance(drones[0]), # distance to drone 0 (including simulated distance measurement noise)
                # ego.position[0], # ego drone x ground-truth position
                # ego.position[1], # ego drone y ground-truth position
                # ego.position[2], # ego drone z ground-truth position
                # drones[0].position[0], # drone 0 x ground-truth position
                # drones[0].position[1], # drone 0 y ground-truth position
                # drones[0].position[2], # drone 0 z ground-truth position
            ),
        )

    def reset_env(self, rng_key, params: EnvParams = None):
        """Resets the environment to its initial state.

        Arguments:
        ---------
        rng_key: The random number generator key.
        params: Env params. unused.

        Returns:
        -------
        - obs: The initial observation.
        - reward: The initial reward.
        - done: A boolean indicating if the episode is done.
        - state: The initial state of the environment.
        """
        k1, k2 = jrandom.split(rng_key)
        initial_state = self.initial_state(k1)
        pos, vel = initial_state["pos"], initial_state["vel"]
        obs, _ = self._sample_observation(pos, vel, k2)
        return obs, initial_state


def run_dronegym(args, out_file: str = "data/dronegym_output.csv"):
    """Run the drone gym simulation and collect data.

    Args:
    ----
        args: An object containing the parameters for the simulation.
        out_file: File for saving the output.

    Returns:
    -------
        A tuple containing the collected data arrays:
        - data_ego_pos: Array of ego drone positions.
        - data_ego_vel: Array of ego drone velocities.
        - data_drones_pos: Array of other drones' positions.
        - data_true_distance: Array of true distances between ego drone and other drones.
        - data_noisy_distance: Array of noisy distances between ego drone and other drones.
        - data_filtered_distance: Array of filtered distances between ego drone and other drones.
    """
    rng_key = jrandom.PRNGKey(0)
    dronegym = DroneGym(EnvParams(args))
    state = dronegym.initial_state(rng_key)

    with open(out_file, "w", newline="") as csvfile:
        # Output file
        pos_writer = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Write header
        pos_writer.writerow(["v_x", "v_y", "v_z", "distance_noisy", "distance_true"])

        data_drones_pos = []
        data_true_distance = []
        data_noisy_distance = []
        data_filtered_distance = []
        data_ego_pos = []
        data_ego_vel = []

        # main simulation, running for args.steps
        for step in range(args.steps):
            if step % 1000 == 0:  # just to print some progress
                print("step: {:3d}".format(step))
            rng_key, step_key = jrandom.split(rng_key)
            action = jrandom.normal(step_key, [dronegym.params.n_dim]) * dronegym.params.change_velocity_stddev
            action = jnp.zeros(dronegym.params.n_dim)
            obs, state, _, _, info = dronegym.step(step_key, state, action)

            data_drones_pos.append(info["data_drones_pos"])
            data_true_distance.append(info["data_true_distance"])
            data_noisy_distance.append(info["data_noisy_distance"])
            data_filtered_distance.append(info["data_filtered_distance"])
            data_ego_pos.append(info["data_ego_pos"])
            data_ego_vel.append(info["data_ego_vel"])

            # get true distance between ego and other drone
            # save data to file
            pos_writer.writerow([obs])

        data_noisy_distance = jnp.vstack(data_noisy_distance)
        data_true_distance = jnp.vstack(data_true_distance)
        data_filtered_distance = jnp.vstack(data_filtered_distance)
        data_ego_pos = jnp.vstack(data_ego_pos)
        data_ego_vel = jnp.vstack(data_ego_vel)
        data_drones_pos = jnp.vstack(data_drones_pos)

        return (
            data_ego_pos,
            data_ego_vel,
            data_drones_pos,
            data_true_distance,
            data_noisy_distance,
            data_filtered_distance,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sim", description="simulate random drone movements and distance changes")
    parser.add_argument("-o", "--output", type=pathlib.Path, default="data/output.csv", help="output file name")
    parser.add_argument("-d", "--dims", choices=[2, 3], default=2, help="number of dimenstions")
    parser.add_argument("-s", "--steps", type=int, default=3000, help="number of simulation steps")
    parser.add_argument("-n", "--ndrones", type=int, default=1, help="number of other drones")
    parser.add_argument(
        "-f", "--frequency", type=int, default=15, help="unit: Hz, used to convert from steps to actual time"
    )
    parser.add_argument("-z", "--noise", type=float, default=0.15, help="distance measurement noise")
    parser.add_argument(
        "-c",
        "--noise_color",
        type=int,
        default=0,
        help="what type of noise should we add? (allowed values: 0=white noise or 2=pink noise A or 3=pink noise B)",
    )
    parser.add_argument(
        "-i", "--iir_filter", type=float, default=0.2, help="iir filter parameter (must be in the range of 0.0 to 1.0)"
    )
    args = parser.parse_args()

    output = run_dronegym(args, out_file=args.output)
