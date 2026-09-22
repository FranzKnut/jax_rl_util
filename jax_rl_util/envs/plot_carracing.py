"""Plotting utilities for the CarRacing environments."""

import argparse
import os
import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

# Zoom factor used by CarRacingPenaltyEnv._render(mode="background")
BACKGROUND_ZOOM = 2.0

DEFAULT_CMAP = LinearSegmentedColormap.from_list(
    "red_white_blue", ["#FF0000", "#FFFFFF", "#0000FF"]
)
SPEED_CMAP = plt.get_cmap("viridis")


def _window_size():
    """Get the render window size of the gymnasium CarRacing env."""
    from gymnasium.envs.box2d.car_racing import WINDOW_H, WINDOW_W

    return WINDOW_W, WINDOW_H


def world_to_pixel(positions):
    """Convert CarRacing world coordinates to background image pixel coordinates.

    Parameters
    ----------
    positions : array_like
        Positions with the last axis holding (x, y) in world coordinates.

    Returns
    -------
    pixels : ndarray
        Positions in pixel coordinates of the background image.
    """
    window_w, window_h = _window_size()
    offset = np.array([window_w / 2, window_h / 2])
    return np.asarray(positions) * BACKGROUND_ZOOM + offset


def make_background(rng=None, env=None, env_name="CarRacingPenalty-v0"):
    """Render the track of a CarRacing env as a background image.

    Parameters
    ----------
    rng : jax.Array, optional
        Key used to reset a freshly created env. Ignored if `env` is given.
    env : optional
        Existing (possibly wrapped) env whose current track is rendered. Using the
        env that generated the trajectories guarantees a matching background.
    env_name : str
        Env id used when creating a fresh env.

    Returns
    -------
    bg_img : ndarray
        Image of the track, already flipped to match `imshow` orientation.
    """
    if env is None:
        from jax_rl_util.envs.environments import EnvironmentConfig, make_wrapped_env

        env, _ = make_wrapped_env(EnvironmentConfig(env_name=env_name))
        env.reset(rng)
    bg_img = np.array(env.unwrapped._render(mode="background"))
    return np.flip(bg_img, axis=0)


def _select_batch(data, batch_index=0, ndim=2):
    """Reduce data to `ndim` axes by indexing trailing batch axes."""
    data = np.asarray(data)
    while data.ndim > ndim:
        data = data[:, batch_index]
    return data


def plot_trajectory(
    ax: Axes,
    positions,
    speeds=None,
    batch_index=0,
    marker="x",
    cmap=SPEED_CMAP,
    norm=None,
    **kwargs,
):
    """Plot a single trajectory in pixel coordinates and mark its end.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    positions : array_like
        World-coordinate positions of shape (T, 2) or (T, batch, ..., 2).
    speeds : array_like, optional
        Per-step speed of shape (T,) or (T, batch, ...). If given, the line is
        colored by speed instead of a single color.
    batch_index : int
        Which batch element to plot for batched positions.
    marker : str
        Marker drawn at the final position.
    cmap : matplotlib.colors.Colormap
        Colormap for speed coloring.
    norm : matplotlib.colors.Normalize, optional
        Speed normalization. Defaults to the range of this trajectory.
    kwargs : any
        Passed to `ax.plot` or to the `LineCollection`.

    Returns
    -------
    artist : matplotlib artist or None
        The line or line collection, None if the trajectory is too short.
    """
    xy = world_to_pixel(_select_batch(positions, batch_index))
    if len(xy) < 2:
        return None
    # Last entry belongs to the reset state already
    xy = xy[:-1]

    if speeds is None:
        (line,) = ax.plot(xy[:, 0], xy[:, 1], **kwargs)
        end_color = line.get_color()
    else:
        speeds = _select_batch(speeds, batch_index, ndim=1)[: len(xy)]
        if norm is None:
            norm = Normalize(vmin=np.min(speeds), vmax=np.max(speeds))
        # Color each segment by the mean speed of its endpoints
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        seg_speeds = (speeds[:-1] + speeds[1:]) / 2
        line = LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
        line.set_array(seg_speeds)
        ax.add_collection(line)
        ax.update_datalim(xy)
        end_color = cmap(norm(speeds[-1]))

    if marker:
        ax.scatter(xy[-1, 0], xy[-1, 1], marker=marker, s=100, color=end_color)
    return line


def plot_carracing(
    trajectories,
    speeds=None,
    bg_img=None,
    ax=None,
    batch_index=0,
    cmap=DEFAULT_CMAP,
    speed_cmap=SPEED_CMAP,
    cbar_label="Finetuning Laps",
    speed_cbar_label="Speed",
    labels=None,
    bg_alpha=0.6,
) -> Figure:
    """Plot CarRacing trajectories on top of the track.

    Parameters
    ----------
    trajectories : array_like or sequence of array_like
        A single trajectory of world-coordinate positions of shape (T, 2) or
        (T, batch, ..., 2), or a list of such trajectories.
    speeds : array_like or sequence of array_like, optional
        Per-step speeds matching `trajectories`. If given, lines are colored by
        speed (shared color scale across trajectories) instead of by index.
    bg_img : ndarray, optional
        Background image as returned by `make_background`.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure is created if omitted.
    batch_index : int
        Which batch element to plot for batched positions.
    cmap : matplotlib.colors.Colormap
        Colormap used to color the trajectories in order.
    speed_cmap : matplotlib.colors.Colormap
        Colormap used when coloring by speed.
    cbar_label : str or None
        Label of the colorbar. No colorbar is drawn if None or for a single trajectory.
    speed_cbar_label : str or None
        Label of the speed colorbar, drawn instead of the index colorbar.
    labels : sequence of str, optional
        Colorbar tick labels. Defaults to the trajectory index starting at 1.
    bg_alpha : float
        Alpha of the background image.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    if not isinstance(trajectories, (list, tuple)):
        # A single array is one trajectory, batched or not
        trajectories = [np.asarray(trajectories)]
    if speeds is not None and not isinstance(speeds, (list, tuple)):
        speeds = [np.asarray(speeds)]
    if speeds is None:
        speeds = [None] * len(trajectories)
    elif len(speeds) != len(trajectories):
        raise ValueError("Need one speed array per trajectory.")

    keep = [i for i, t in enumerate(trajectories) if len(t) > 1]
    if not keep:
        raise ValueError("No trajectory with more than one position given.")
    trajectories = [trajectories[i] for i in keep]
    speeds = [speeds[i] for i in keep]
    by_speed = speeds[0] is not None

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ticks = np.linspace(0, 1, len(trajectories))
    if by_speed:
        # Shared scale so colors are comparable across trajectories
        norm = Normalize(
            vmin=min(np.min(s) for s in speeds), vmax=max(np.max(s) for s in speeds)
        )
        colors = [None] * len(trajectories)
    else:
        norm = None
        colors = cmap(ticks) if len(trajectories) > 1 else [cmap(1.0)]

    for color, positions, speed in zip(colors, trajectories, speeds):
        plot_trajectory(
            ax,
            positions,
            speeds=speed,
            batch_index=batch_index,
            cmap=speed_cmap,
            norm=norm,
            **({} if by_speed else {"color": color}),
        )

    ax.axis("off")
    if bg_img is not None:
        ax.imshow(bg_img, alpha=bg_alpha)
    else:
        ax.autoscale_view()

    cbar_kwargs = dict(
        orientation="horizontal", shrink=0.6, ax=ax, pad=0.02, fraction=0.046
    )
    if by_speed and speed_cbar_label is not None:
        fig.colorbar(
            plt.cm.ScalarMappable(cmap=speed_cmap, norm=norm),
            label=speed_cbar_label,
            **cbar_kwargs,
        )
    elif not by_speed and cbar_label is not None and len(trajectories) > 1:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap), label=cbar_label, **cbar_kwargs
        )
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(
            labels
            if labels is not None
            else [str(i + 1) for i in range(len(trajectories))]
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.08)
    fig.tight_layout(pad=0)
    return fig


def _step_number(file_name):
    """Get the trailing step number of a saved array file, for natural sorting."""
    digits = "".join(re.findall(r"\d+", os.path.splitext(file_name)[0])[-1:])
    return int(digits) if digits else 0


def plot_from_files(
    artifact_path,
    rng=None,
    prefix="eval_",
    max_trajectories=5,
    env_name="CarRacingPenalty-v0",
    color_by_speed=False,
    **kwargs,
) -> Figure:
    """Plot trajectories stored as .npy files in a directory.

    Expects positions in `{prefix}pos_{step}.npy` and, for speed coloring,
    matching `{prefix}speed_{step}.npy` files.

    Parameters
    ----------
    artifact_path : str
        Directory holding the saved arrays.
    rng : jax.Array, optional
        Key used to reset the env that renders the background.
    prefix : str
        Only files starting with this prefix are plotted.
    max_trajectories : int or None
        Plot at most this many trajectories (ordered by step number).
    env_name : str
        Env id used for the background.
    color_by_speed : bool
        Color by speed if a speed file exists for every plotted trajectory.
    kwargs : any
        Passed to `plot_carracing`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    files = [
        f
        for f in os.listdir(artifact_path)
        if f.startswith(prefix) and "speed" not in f and f.endswith(".npy")
    ]
    files = sorted(files, key=_step_number)
    if max_trajectories is not None:
        files = files[:max_trajectories]

    trajectories = [np.load(os.path.join(artifact_path, f)) for f in files]

    speeds = None
    if color_by_speed:
        speed_files = [
            os.path.join(artifact_path, f.replace("pos", "speed")) for f in files
        ]
        if all(os.path.exists(f) for f in speed_files):
            speeds = [np.load(f) for f in speed_files]
        else:
            print("No speed files found, coloring by trajectory index.")

    bg_img = make_background(rng, env_name=env_name)
    return plot_carracing(trajectories, speeds=speeds, bg_img=bg_img, **kwargs)


if __name__ == "__main__":
    import jax.random as jrandom

    parser = argparse.ArgumentParser(
        prog="plot_carracing", description="plot CarRacing trajectories on the track"
    )
    parser.add_argument("artifact_path", help="directory with saved position arrays")
    parser.add_argument("--prefix", default="eval_", help="file name prefix")
    parser.add_argument("--seed", type=int, default=0, help="seed for the background")
    parser.add_argument("--max", type=int, default=5, help="max number of trajectories")
    parser.add_argument("-o", "--out", default=None, help="output file name")
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="color by trajectory index instead of speed",
    )
    args = parser.parse_args()

    _key, key_env = jrandom.split(jrandom.PRNGKey(args.seed))
    fig = plot_from_files(
        args.artifact_path,
        rng=key_env,
        prefix=args.prefix,
        max_trajectories=args.max,
        color_by_speed=not args.no_speed,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, bbox_inches="tight")
    plt.show()
