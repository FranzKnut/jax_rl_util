"""Plotting functions. Each takes a DataFrame and returns the matplotlib figure."""

import os
import re
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from jax_rl_util.eval.select import group_label, ordered

PLOT_DIR = "plots"


def save_fig(fig, name: str, out_dir: str = PLOT_DIR, formats=("pdf",), transparent: bool = False) -> list[str]:
    """Save a figure as ``<out_dir>/<name>.<fmt>`` for each format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File stem; characters that are unsafe in file names become ``_``.
    out_dir : str, optional
        Directory, by default "plots".
    formats : tuple[str, ...], optional
        File formats, by default ("pdf",).
    transparent : bool, optional
        Transparent background, by default False.

    Returns
    -------
    list[str]
        Written paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = re.sub(r"(?u)[^-\w.\[\],]", "_", name)
    paths = []
    for f in formats:
        p = os.path.join(out_dir, f"{stem}.{f}")
        fig.savefig(p, transparent=transparent, bbox_inches="tight")
        paths.append(p)
        print("Saved", p)
    return paths


def _subplots(n: int, figsize=None, rows: int = 1, sharey: bool = False):
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (max(3.0, 3 * cols), 3 * rows), sharey=sharey, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _splits(df, split_by, split_order=None):
    if not split_by:
        return [("", df)]
    keys = ordered(df[split_by].dropna().unique(), split_order)
    return [(k, df[df[split_by] == k]) for k in keys]


def _finish(fig, axes, title="", ylim=None, xlim=None, ylabel="", xlabel="", logy=False):
    for ax in axes:
        if ylim:
            ax.set_ylim(*ylim)
        if xlim:
            ax.set_xlim(*xlim)
        if logy:
            ax.set_yscale("log")
    if ylabel:
        axes[0].set_ylabel(ylabel)
    if xlabel:
        for ax in axes:
            ax.set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def boxplot(
    df: pd.DataFrame,
    value: str,
    by,
    split_by: str = "",
    order: list | None = None,
    split_order: list | None = None,
    label_fn: Callable[[pd.Series], str] | None = None,
    rows: int = 1,
    figsize=None,
    sharey: bool = False,
    rot: float = 30,
    **finish_kw,
):
    """Box plot of ``value`` per ``by`` group, one subplot per ``split_by`` value.

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    value : str
        Column to plot.
    by : str or list[str]
        Columns that define the boxes; joined into one label.
    split_by : str, optional
        Column with one subplot per value, by default none.
    order, split_order : list, optional
        Preferred order of box labels / subplots.
    label_fn : Callable, optional
        Row -> tick label, overriding the joined ``by`` values.
    rows : int, optional
        Subplot rows, by default 1.
    figsize : tuple, optional
        Figure size.
    sharey : bool, optional
        Share the y axis, by default False.
    rot : float, optional
        Tick label rotation, by default 30.
    **finish_kw
        ``title``, ``ylim``, ``xlim``, ``ylabel``, ``xlabel``, ``logy``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.assign(_label=df.apply(label_fn, axis=1) if label_fn else group_label(df, by))
    splits = _splits(df, split_by, split_order)
    fig, axes = _subplots(len(splits), figsize, rows, sharey)
    for ax, (key, d) in zip(axes, splits, strict=True):
        labels = ordered(d["_label"].unique(), order)
        ax.boxplot([d.loc[d["_label"] == lab, value].dropna().to_numpy() for lab in labels])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
        ax.set_title(str(key))
        ax.grid(alpha=0.3)
    return _finish(fig, axes, **finish_kw)


def barplot(
    df: pd.DataFrame,
    value: str,
    by,
    split_by: str = "",
    agg: str = "mean",
    err: str = "std",
    order: list | None = None,
    split_order: list | None = None,
    colors: dict | None = None,
    rows: int = 1,
    figsize=None,
    sharey: bool = False,
    rot: float = 90,
    **finish_kw,
):
    """Bar plot of ``agg(value)`` per ``by`` group with optional error bars.

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    value : str
        Column to plot.
    by : str or list[str]
        Columns that define the bars.
    split_by : str, optional
        Column with one subplot per value.
    agg : str, optional
        Bar height, by default "mean".
    err : str, optional
        Error bar: "std", "sem" or "" for none, by default "std".
    order, split_order : list, optional
        Preferred order of bars / subplots.
    colors : dict, optional
        Label -> color. By default the tableau cycle.
    rows, figsize, sharey, rot
        Layout options.
    **finish_kw
        ``title``, ``ylim``, ``xlim``, ``ylabel``, ``xlabel``, ``logy``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.assign(_label=group_label(df, by))
    splits = _splits(df, split_by, split_order)
    fig, axes = _subplots(len(splits), figsize, rows, sharey)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (key, d) in zip(axes, splits, strict=True):
        g = d.groupby("_label")[value]
        labels = ordered(g.groups.keys(), order)
        heights = g.agg(agg).reindex(labels)
        yerr = g.agg(err).reindex(labels) if err else None
        c = [(colors or {}).get(lab, cycle[i % len(cycle)]) for i, lab in enumerate(labels)]
        ax.bar(labels, heights, yerr=yerr, color=c, capsize=3 if err else 0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
        ax.set_title(str(key))
    return _finish(fig, axes, **finish_kw)


def aggregate_curves(
    df: pd.DataFrame,
    x: str = "step",
    value: str = "value",
    run: str = "run_id",
    agg: str = "mean",
    shade: str = "std",
    window: int = 0,
    downsample: int = 1,
) -> pd.DataFrame:
    """Aggregate the curves of several runs into center and band.

    Parameters
    ----------
    df : pd.DataFrame
        Long table for one line: columns ``x``, ``value`` and ``run``. If it has
        ``min`` / ``max`` columns (wandb exports), those are used for the band
        when ``shade == "minmax"``.
    x : str, optional
        Step column, by default "step".
    value : str, optional
        Value column, by default "value".
    run : str, optional
        Run id column, by default "run_id". Without it every row is one run.
    agg : str, optional
        Center: "mean" or "median", by default "mean".
    shade : str, optional
        Band: "std", "var", "sem" (center ± stat), "minmax" or "" for none.
    window : int, optional
        Rolling mean window applied to center and band, by default 0 (off).
    downsample : int, optional
        Keep every n-th step, by default 1.

    Returns
    -------
    pd.DataFrame
        Indexed by ``x`` with columns ``center``, ``lo``, ``hi``.
    """
    g = df.groupby(x)[value]
    out = pd.DataFrame({"center": g.agg(agg)})
    if shade == "minmax":
        out["lo"] = df.groupby(x)["min"].min() if "min" in df else g.min()
        out["hi"] = df.groupby(x)["max"].max() if "max" in df else g.max()
    elif shade:
        s = g.agg(shade).fillna(0)
        out["lo"], out["hi"] = out["center"] - s, out["center"] + s
    out = out.sort_index().iloc[::downsample]
    if window and window > 1:
        out = out.rolling(window, min_periods=1).mean()
    return out


def lineplot(
    df: pd.DataFrame,
    by,
    x: str = "step",
    value: str = "value",
    run: str = "run_id",
    split_by: str = "",
    agg: str = "mean",
    shade: str = "std",
    window: int = 0,
    downsample: int = 1,
    x_scale: float = 1.0,
    events: list[float] | None = None,
    order: list | None = None,
    split_order: list | None = None,
    colors: dict | None = None,
    linestyles: dict | None = None,
    legend_title: str = "",
    legend_loc: str = "best",
    alpha: float = 0.3,
    rows: int = 1,
    figsize=None,
    sharey: bool = False,
    **finish_kw,
):
    """Learning curves: one line per ``by`` group, aggregated over runs.

    Parameters
    ----------
    df : pd.DataFrame
        Long history table (see ``data.load_history``).
    by : str or list[str]
        Columns that define the lines.
    x, value, run : str, optional
        Step, value and run id columns.
    split_by : str, optional
        Column with one subplot per value.
    agg, shade, window, downsample
        See ``aggregate_curves``.
    x_scale : float, optional
        Multiply steps by this factor (e.g. 1e-3 for thousands), by default 1.
    events : list[float], optional
        Steps (after scaling) where a dashed vertical line is drawn, e.g. the
        start of a perturbation.
    order, split_order : list, optional
        Preferred order of lines / subplots.
    colors, linestyles : dict, optional
        Label -> color / line style.
    legend_title : str, optional
        Legend title.
    legend_loc : str, optional
        Legend location on the last subplot, "figure" for one figure legend at
        the bottom right, "" for no legend.
    alpha : float, optional
        Band opacity, by default 0.3.
    rows, figsize, sharey
        Layout options.
    **finish_kw
        ``title``, ``ylim``, ``xlim``, ``ylabel``, ``xlabel``, ``logy``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df.assign(_label=group_label(df, by))
    if run not in df.columns:
        df = df.assign(**{run: np.arange(len(df))})
    splits = _splits(df, split_by, split_order)
    fig, axes = _subplots(len(splits), figsize, rows, sharey)
    for ax, (key, d) in zip(axes, splits, strict=True):
        for lab in ordered(d["_label"].unique(), order):
            c = aggregate_curves(d[d["_label"] == lab], x, value, run, agg, shade, window, downsample)
            xs = c.index.to_numpy(dtype=float) * x_scale
            kw = {}
            if colors and lab in colors:
                kw["color"] = colors[lab]
            if linestyles and lab in linestyles:
                kw["linestyle"] = linestyles[lab]
            (line,) = ax.plot(xs, c["center"], label=lab, **kw)
            if shade:
                ax.fill_between(xs, c["lo"], c["hi"], alpha=alpha, color=line.get_color(), linewidth=0)
        for e in events or []:
            ax.axvline(e, color="gray", linestyle="--", linewidth=1)
        ax.set_title(str(key))
        ax.grid(alpha=0.3)
    if legend_loc == "figure":  # one legend for the whole figure, bottom right
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, title=legend_title or None, loc="lower right")
    elif legend_loc:
        axes[-1].legend(title=legend_title or None, loc=legend_loc)
    return _finish(fig, axes, **finish_kw)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str = "",
    marker_by: str = "",
    markers: dict | None = None,
    cmap: str = "viridis",
    figsize=None,
    legend_title: str = "",
    colorbar_label: str = "",
    **finish_kw,
):
    """Scatter plot with optional numeric coloring and one marker per category.

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    x, y : str
        Axis columns.
    color_by : str, optional
        Numeric column mapped to colors (with a colorbar).
    marker_by : str, optional
        Categorical column with one marker each.
    markers : dict, optional
        Category -> marker, by default cycles through "o", "x", "^", "s", "D".
    cmap : str, optional
        Colormap, by default "viridis".
    figsize : tuple, optional
        Figure size.
    legend_title, colorbar_label : str, optional
        Labels.
    **finish_kw
        ``title``, ``ylim``, ``xlim``, ``ylabel``, ``xlabel``, ``logy``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize or (5, 3))
    cats = ordered(df[marker_by].dropna().unique()) if marker_by else [None]
    default = ["o", "x", "^", "s", "D", "v", "*"]
    markers = {c: (markers or {}).get(c, default[i % len(default)]) for i, c in enumerate(cats)}
    vmin = df[color_by].min() if color_by else None
    vmax = df[color_by].max() if color_by else None
    sc = None
    for cat in cats:
        d = df if cat is None else df[df[marker_by] == cat]
        sc = ax.scatter(
            d[x], d[y], c=d[color_by] if color_by else None, cmap=cmap if color_by else None,
            vmin=vmin, vmax=vmax, marker=markers[cat], label=str(cat),
        )
    if color_by and sc is not None:
        fig.colorbar(sc, ax=ax, label=colorbar_label or color_by)
    if marker_by:
        handles = [
            Line2D([0], [0], marker=markers[c], color="w", markerfacecolor="darkblue", markeredgecolor="darkblue",
                   markersize=6, label=str(c))
            for c in cats
        ]
        ax.legend(handles=handles, title=legend_title or None)
    finish_kw["xlabel"] = finish_kw.get("xlabel") or x
    finish_kw["ylabel"] = finish_kw.get("ylabel") or y
    return _finish(fig, [ax], **finish_kw)
