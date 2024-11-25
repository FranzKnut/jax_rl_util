#!/usr/bin/env python3
"""A gym environment for drone simulations."""

import argparse
import csv
import pathlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import plotly.graph_objects as go


@dataclass
class DroneGymParams:
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

    output: str = "data/output.csv"  # Output file name
    frequency: int = 15
    dims: int = 2
    ndrones: int = 1
    frequency: int = 15

    # velocity parameters for ego drone
    initial_velocity_stddev: float = 0.1
    ego_change_velocity_stddev: float = 0.005

    # velocity parameters for other drones
    velocity_stddev: float = 0
    change_velocity_stddev: float = 0

    # distance measurement noise
    noise_stddev: float = 0.15
    noise_color: int = 0
    # initial position distribution for other drones
    initial_pos_stddev: float = 5
    iir_filter: float = 0.2
    noise_iir_value: float = 0
    goto_stddev: float = 25.0

    n_drones: int = 1
    n_dim: int = 2
    starting_post_ego: Tuple[float, float, float] = (0, 0, 0)
    plot_range: int = 15
    steps: int = 3000


class DroneGym:
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

    def __init__(self, params: DroneGymParams):
        """Initialize the DroneGym object.

        Arguments:
        ---------
        params: A `DroneGymParams` object containing the parameters for the environment.
        """
        # initialize empty arrays
        self._step = 0
        assert params.n_dim in [2, 3], "Only 2D and 3D is supported"
        self.params = params

        # initialize ego and other drones
        self.starting_pos_ego = jnp.array(params.starting_post_ego)[: params.n_dim]

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
                self.starting_pos_ego[None],
                self.params.initial_pos_stddev * jrandom.normal(k_pos, [self.params.n_drones, self.params.n_dim]),
            ],
            axis=0,
        )
        difference = initial_pos[0] - initial_pos[1:]
        true_distance = jnp.linalg.norm(difference, axis=-1)
        return OrderedDict(
            {
                "step": 0,
                "pos": initial_pos,
                "vel": jnp.zeros([self.params.n_drones + 1, self.params.n_dim]),
                "goto": jnp.zeros([self.params.n_drones, self.params.n_dim]),
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

    def step(self, state, action=None):
        """Perform a step in the environment.

        Arguments:
        ---------
        state: The current state of the environment.
        action: The action to be performed (optional).

        Returns:
        -------
        - obs: The observation after the step.
        - state: The updated state of the environment.
        - reward: The reward obtained from the step.
        - done: A boolean indicating if the episode is done.
        - info: Additional information about the step.
        """
        # perform ego and other drone movement for next step
        step, pos, vel, goto, filtered_dist, key = state.values()
        ego_key, obs_key, drone_key, next_key = jrandom.split(key, 4)
        ego_pos = pos[0]
        ego_vel = vel[0]

        # Random movement for other drones
        drones_acc = (
            jrandom.normal(drone_key, [self.params.n_drones, self.params.n_dim]) * self.params.change_velocity_stddev
        )
        drones_vel = vel[1:] + drones_acc

        # controlled movement for ego drone
        # controlled_movement_a
        #     vel = vel + jrandom.normal(rng_key, [self.params.n_dim]) * self.params.change_velocity_stddev

        #     lim_stddev = 15
        #     lim_factor = 500
        #     jnp.where(jnp.abs(pos) > jrandom.normal(0.0, lim_stddev), vel[0] -= vel[0] / lim_factor, vel[0])
        #     if (abs(self.params.position[0]) > jrandom.normal(0.0, lim_stddev)):
        #         vel[0] -= vel[0] / lim_factor
        #     if (abs(self.position[1]) > jrandom.normal(0.0, lim_stddev)):
        #         self.velocity[1] -= self.position[1] / lim_factor
        #     if (abs(self.position[2]) > jrandom.normal(0.0, lim_stddev)):
        #         self.velocity[2] -= self.position[2] / lim_factor

        # controlled_movement_b
        acc_key, go_key = jrandom.split(ego_key)
        ego_vel += jrandom.normal(acc_key, [self.params.n_dim]) * self.params.change_velocity_stddev
        goto = ego_vel + jnp.where(
            step % 10 == 0, jrandom.normal(go_key, [self.params.n_dim]) * self.params.goto_stddev, goto
        )
        ego_vel = ego_vel * 0.99 + (goto - ego_pos) * 0.0004

        # Concatenate ego and drones velocities
        vel = jnp.concatenate([ego_vel, drones_vel], axis=0)
        pos += vel * 1 / self.params.frequency

        # Sample observation
        obs = self._sample_observation(pos, vel, obs_key)
        data_noisy_distance = obs[-2]

        filtered_dist = filtered_dist * (1.0 - self.params.iir_filter) + data_noisy_distance * (self.params.iir_filter)

        self._step += 1

        state = OrderedDict(
            {"step": step + 1, "pos": pos, "vel": vel, "goto": goto, "filtered_dist": filtered_dist, "rng": next_key}
        )

        return (
            obs,
            state,
            0,
            False,
            {
                "data_ego_pos": state["pos"][0],
                "data_ego_vel": state["vel"][0],
                "data_drones_pos": state["pos"][1:],
                "data_true_distance": obs[-1],
                "data_noisy_distance": data_noisy_distance,
                "data_filtered_distance": filtered_dist,
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
        difference = pos[0] - pos[1:]
        true_distance = jnp.linalg.norm(difference, axis=-1)
        noisy_distance = self.apply_noise(true_distance, rng_key)

        return (
            vel[0],  # ego drone velocity
            # Prediction target:
            jnp.concatenate(
                [
                    pos[0],  # ego drone position
                    # noisy_distance,  # noisy distance to drone 0
                ]
            ),
            noisy_distance,  # noisy distance to drone 0
            true_distance,  # true distance to drone 0
            # ego.get_distance(drones[0]), # distance to drone 0 (including simulated distance measurement noise)
            # ego.position[0], # ego drone x ground-truth position
            # ego.position[1], # ego drone y ground-truth position
            # ego.position[2], # ego drone z ground-truth position
            # drones[0].position[0], # drone 0 x ground-truth position
            # drones[0].position[1], # drone 0 y ground-truth position
            # drones[0].position[2], # drone 0 z ground-truth position
        )

    def reset(self, rng_key):
        """Resets the environment to its initial state.

        Arguments:
        ---------
        rng_key: The random number generator key.

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
        return self._sample_observation(pos, vel, k2), 0, False, initial_state


def run_dronegym(args):
    """Run the drone gym simulation and collect data.

    Args:
    ----
        args: An object containing the parameters for the simulation.

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
    dronegym = DroneGym(DroneGymParams(args))
    state = dronegym.initial_state(rng_key)

    with open(args.output, "w", newline="") as csvfile:
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
            obs, state, _, _, info = dronegym.step(state)

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


def plot_drone_pos(ax, data_ego, data_drones, plot_range=15):
    """Plot the position of the drone on the given axis.

    Arguments:
    ---------
    ax (matplotlib.axes.Axes): The axis on which to plot the drone position.
    data_ego (tuple): A tuple containing the x and y coordinates of the ego drone.
    data_drones (tuple): A tuple containing the x and y coordinates of the other drones.
    plot_range (float, optional): The range of the plot. Defaults to 15.

    Returns:
    -------
    None
    """
    data_ego_x, data_ego_y = data_ego
    data_drones_x, data_drones_y = data_drones
    ax.plot(data_ego_x, data_ego_y, color="red")
    ax.plot(data_ego_x[0], data_ego_y[0], "rx")
    ax.plot(data_drones_x, data_drones_y, color="blue")
    ax.plot(data_drones_x[0], data_drones_y[0], "bx")
    ax.set_xlim([-plot_range, plot_range])
    ax.set_ylim([-plot_range, plot_range])


def plotly_drone_pos(fig, data_drones_pos, name):
    """Plot the drone positions with plotly."""
    # Add scatter plot for drone positions
    fig.add_trace(go.Scatter(x=data_drones_pos[..., 0], y=data_drones_pos[..., 1], mode="markers", name=name))

    # if args.dims == 3:
    #     # Add scatter plot for drone positions in 3D
    #     fig.add_trace(go.Scatter(x=data_ego_pos[..., 2], y=data_drones_pos[..., 1], mode='markers', name='Ego Drone'), row=1, col=2)
    #     fig.add_trace(go.Scatter(x=data_ego_pos[..., 0], y=data_drones_pos[..., 2], mode='markers', name='Other Drones'), row=2, col=1)


def plot_drones(
    args: DroneGymParams,
    data_ego_pos,
    data_ego_vel,
    data_drones_pos,
    data_true_distance,
    data_noisy_distance,
    data_filtered_distance,
):
    """Plot the drone positions and distances."""
    # We transpose so that dim 1 is the time axis and dim 2 is the drone index
    data_noisy_distance = data_noisy_distance.T
    data_true_distance = data_true_distance.T
    data_filtered_distance = data_filtered_distance.T

    # Make subplots
    _, axes = plt.subplots(2 if args.dims == 3 else 1, 2)

    # time axis
    data_time = jnp.linspace(0, (args.steps - 1) / args.frequency, num=args.steps)

    ax0 = axes[0] if args.dims == 2 else axes[0, 0]
    ax3 = axes[-1] if args.dims == 2 else axes[-1, -1]

    data_ego = (data_ego_pos[..., 0], data_ego_pos[..., 1])
    data_drones = (data_drones_pos[..., 0], data_drones_pos[..., 1])
    plot_drone_pos(ax0, data_ego, data_drones)
    ax0.set_xlabel("X position [m]")
    ax0.set_ylabel("Y position [m]")

    if args.dims == 3:
        ax1, ax2 = axes[0, 1], axes[1, 0]
        plot_drone_pos(ax1, data_ego_pos[..., 2], data_drones_pos[..., 1])
        ax1.set_xlabel("Z position [m]")
        ax1.set_ylabel("Y position [m]")
        plot_drone_pos(ax2, data_ego_pos[..., 0], data_drones_pos[..., 2])
        ax2.set_xlabel("X position [m]")
        ax2.set_ylabel("Z position [m]")

    plot_distances(
        args,
        ax3,
        data_time,
        {
            "Ground-truth distance": data_true_distance,
            "Noisy measurement": data_noisy_distance,
            "IIR-Filtered measurement": data_filtered_distance,
        },
        data_ego_vel,
    )

    plt.savefig("plots/drone_sim.png")


def plot_distances(args: DroneGymParams, ax, data_time, data: dict, data_ego_vel=None):
    """Plot distances between drones and ego velocity over time.

    Args:
    ----
        args (DroneGymParams): Parameters for the drone gym environment.
        ax: The matplotlib axes object to plot on.
        data_time: The time values for the data.
        data (dict): A dictionary containing the distances between drones.
        data_ego_vel: The ego velocity data.

    Returns:
    -------
        None
    """
    for name, distance in data.items():
        for i in range(args.ndrones):
            ax.plot(data_time, distance[i], label=name + " drone {:d}".format(i))

    if data_ego_vel is not None:
        ax.plot(data_time, data_ego_vel[:, 0], color="green")
        ax.plot(data_time, data_ego_vel[:, 1], color="orange")
        if args.dims == 3:
            ax.plot(data_time, data_ego_vel[:, 2], color="yellow")
    ax.set_ylim([0, 1.41 * args.plot_range])
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")


def plotly_drone_eval(fig: go.Figure, eval_output, gym_params: DroneGymParams):
    """Plot the evaluation results for drone distance estimation using plotly.

    Args:
    ----
        fig: The to plot into
        eval_output (dict): The evaluation output containing the data and predictions.
                            Output of run_eval_episode() in training_util.py.
        gym_params: params of the DroneGym

    Returns:
    -------
        None
    """
    losses, infos, infos_pred, (model, wm_state, opt_state) = eval_output

    # Concatenate metrics
    lines_data = {
        "Ground-truth distance": jnp.concatenate([infos["data_true_distance"], infos_pred["data_true_distance"]]).T,
        "Noisy measurement": jnp.concatenate([infos["data_noisy_distance"], infos_pred["data_noisy_distance"]]).T,
        "IIR-Filtered measurement": jnp.concatenate(
            [infos["data_filtered_distance"], infos_pred["data_filtered_distance"]]
        ).T,
        "RNN prediction": jnp.concatenate([infos["pred"][..., None, -1], infos_pred["pred"][..., -1]]).T,
    }

    # Assuming number of steps is the same for all
    observe_steps = infos["data_true_distance"].shape[0]
    predict_steps = infos_pred["data_true_distance"].shape[0]
    total_steps = observe_steps + predict_steps
    data_time = jnp.linspace(0, total_steps / gym_params.frequency, num=total_steps)

    for line_name, line_data in lines_data.items():
        for drone_idx in range(len(line_data)):
            fig.add_scatter(
                x=data_time, y=line_data[drone_idx], mode="lines", name=line_name + " drone {:d}".format(drone_idx)
            )

    fig.add_scatter(
        x=data_time, y=jnp.concatenate([infos["mae"], infos_pred["mae"]]), mode="lines", name="Absolute error"
    )

    fig.add_shape(
        type="line",
        x0=observe_steps / gym_params.frequency,
        y0=jnp.min(lines_data["RNN prediction"]),
        x1=observe_steps / gym_params.frequency,
        y1=jnp.max(lines_data["RNN prediction"]),
        line=dict(color="Black", width=2, dash="dash"),
    )


def plot_drone_eval(ax, eval_output, gym_params: DroneGymParams):
    """Plot the evaluation results for drone distance estimation.

    Args:
    ----
        ax (matplotlib.axes.Axes): The axes object to plot on.
        eval_output (dict): The evaluation output containing the data and predictions.
                            Output of run_eval_episode() in training_util.py.
        gym_params: DroneGym params

    Returns:
    -------
        None
    """
    losses, infos, infos_pred, (model, wm_state, opt_state) = eval_output

    # Concatenate metrics
    lines_data = {
        "Ground-truth distance": jnp.concatenate([infos["data_true_distance"], infos_pred["data_true_distance"]]).T,
        "Noisy measurement": jnp.concatenate([infos["data_noisy_distance"], infos_pred["data_noisy_distance"]]).T,
        "IIR-Filtered measurement": jnp.concatenate(
            [infos["data_filtered_distance"], infos_pred["data_filtered_distance"]]
        ).T,
        "RNN prediction": jnp.concatenate([infos["pred"], infos_pred["pred"][:, 0]]).T,
    }

    # Assuming number of steps is the same for all
    observe_steps = infos["data_true_distance"].shape[0]
    predict_steps = infos_pred["data_true_distance"].shape[0]
    total_steps = observe_steps + predict_steps
    data_time = jnp.linspace(0, total_steps / gym_params.frequency, num=total_steps)

    plot_distances(gym_params, ax, data_time, lines_data)
    # MAE is not per-drone
    ax.plot(data_time, jnp.concatenate([infos["mae"], infos_pred["mae"]]), label="Absolute error")
    # Horizontal line at start of prediction
    ax.vlines(
        observe_steps / gym_params.frequency,
        ymin=jnp.min(lines_data["Ground-truth distance"]),
        ymax=jnp.max(lines_data["Ground-truth distance"]),
        color="k",
        linestyle="--",
    )

    ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sim", description="simulate random drone movements and distance changes")
    parser.add_argument("-o", "--output", type=pathlib.Path, default="data/output.csv", help="output file name")
    parser.add_argument("-d", "--dims", choices=[2, 3], default=2, help="number of dimenstions")
    parser.add_argument("-s", "--steps", type=int, default=3000, help="number of simulation steps")
    parser.add_argument("-n", "--ndrones", type=int, default=1, help="number of other drones")
    parser.add_argument(
        "-f", "--frequency", type=int, default=15, help="unit: Hz, used to convert from steps to actual time"
    )
    parser.add_argument("-p", "--plot", type=int, default=1, help="should we plot the data? (allowed values: 0 or 1)")
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

    output = run_dronegym(args)
    plot_drones(DroneGymParams(args), *output)
