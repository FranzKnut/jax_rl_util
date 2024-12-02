"""Plotting utilities for dronegym."""

import argparse
import os
import pathlib
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from gymnax import EnvParams
from matplotlib.axes import Axes


def plot_drone_pos(ax: Axes, data_ego, data_drones, plot_range=15):
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
    ax.plot(data_ego_x[-1], data_ego_y[-1], "rx")
    circle = plt.Circle((data_drones_x[-1], data_drones_y[-1]), 0.5, color="blue", fill=None)
    ax.add_artist(circle)
    ax.plot(data_drones_x, data_drones_y, color="blue")
    # ax.plot(data_drones_x[0], data_drones_y[0], "bx")
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


def plot_drones(args: EnvParams, data_ego_pos, data_ego_vel, data_drones_pos, data_true_distance):
    """Plot the drone positions and distances."""
    # data_noisy_distance = data["data_noisy_distance"]
    # data_filtered_distance = data["data_filtered_distance"]

    # We transpose so that dim 1 is the time axis and dim 2 is the drone index
    # data_noisy_distance = data_noisy_distance.T
    data_true_distance = data_true_distance.T
    # data_filtered_distance = data_filtered_distance.T

    # Make subplots
    _, axes = plt.subplots(2 if args.dims == 3 else 1, 2)

    # time axis
    steps = data_ego_pos.shape[0]
    data_time = jnp.linspace(0, (steps - 1) / args.frequency, num=steps)

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
            # "Noisy measurement": data_noisy_distance,
            # "IIR-Filtered measurement": data_filtered_distance,
        },
        data_ego_vel,
    )

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/drone_sim.png")


def plot_distances(args: EnvParams, ax, data_time, data: dict, data_ego_vel=None):
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


def plotly_drone_eval(fig: go.Figure, eval_output, gym_params: EnvParams):
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


def plot_drone_eval(ax, eval_output, gym_params: EnvParams):
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


def plot_from_file(file="data/ppo_best_trajectory.npz", args_file="data/ppo_env_params.pkl", plot_which="best"):
    """Plot the evaluation results from a file.

    Args:
    ----
        file (str, optional): The file to load the evaluation results from. Defaults to "data/ppo_best_trajectory.npz".
        args_file (str, optional): The file to load the gym parameters from. Defaults to "data/ppo_env_params.pkl".
        plot_which (str, optional): Which episode to plot. Defaults to "best".

    Returns:
    -------
        None
    """
    data = np.load(file)
    if os.path.exists(args_file):
        with open(args_file, "rb") as f:
            params = pickle.load(f)
    else:
        params = EnvParams()

    ep_ends = np.argmax(data["done"], axis=0)
    ep_rewards = np.cumsum(data["reward"], axis=0)[ep_ends, jnp.arange(len(ep_ends))]
    idx_selected = np.argmax(ep_rewards) if plot_which == "best" else np.argmin(ep_rewards)
    _ep_end = ep_ends[idx_selected]
    if _ep_end == 0:
        _ep_end = -1
    print("Best ep:", idx_selected, " (%f Reward)" % ep_rewards[idx_selected])
    data_ego_pos = data["pos"][:_ep_end, idx_selected, 0, :]
    data_ego_vel = data["vel"][:_ep_end, idx_selected, 0, :]
    data_drones_pos = data["pos"][:_ep_end, idx_selected, 1:, :]
    data_true_distance = data["reward"][:_ep_end, idx_selected]
    plot_drones(params, data_ego_pos, data_ego_vel, data_drones_pos, data_true_distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sim", description="simulate random drone movements and distance changes")
    parser.add_argument(
        "-f", "--file", type=pathlib.Path, default="data/ppo_best_trajectory.npz", help="output file name"
    )
    parser.add_argument("--args_file", type=pathlib.Path, default="data/ppo_env_params.pkl", help="output file name")
    args = parser.parse_args()

    plot_from_file(args.file, args.args_file, "best")
    plt.show()
