"""EvalConfig, the ``run`` pipeline and the command line entry point.

A repo describes each plot or table as an ``EvalConfig`` preset and calls
``main(PRESETS, derive=derive)`` from its ``<package>/eval.py``::

    python -m rtrrl.eval --list
    python -m rtrrl.eval boxplot_popgym --show
    python -m rtrrl.eval boxplot_popgym --query "seed < 4" --out boxplot_3seeds
"""

import dataclasses
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import simple_parsing
from simple_parsing import field

from jax_rl_util.eval import data as D
from jax_rl_util.eval import plots as P
from jax_rl_util.eval import select as S
from jax_rl_util.eval import tables as T

Kind = Literal["box", "bar", "line", "scatter", "table", "summary", "download"]


@dataclass
class EvalConfig:
    """One plot or table, from data source to output file."""

    kind: Kind = "box"  # what to make
    name: str = ""  # preset name, default output file stem

    # Data source: explicit files, or a wandb project (cached in data_dir)
    data: list[str] = field(default_factory=list)  # run or history files, dirs or globs
    project: str = ""  # wandb project, "name" or "entity/name"
    sweeps: list[str] = field(default_factory=list)  # sweep ids, empty for the whole project
    history_keys: list[str] = field(default_factory=list)  # metrics to download for line plots
    x_key: str = "_step"  # history step key on wandb
    entity: str = D.ENTITY
    data_dir: str = D.DATA_DIR
    download: bool = False  # re-download even if cached
    wandb_filters: dict = field(default_factory=dict, cmd=False)
    loader: Callable[["EvalConfig"], pd.DataFrame] | None = field(default=None, cmd=False)  # custom data loader

    # Columns and selection
    fields: list[str] = field(default_factory=list)  # config fields to pull, "name" or "alias=path"
    rename: dict = field(default_factory=dict, cmd=False)  # {column: {old: new}}
    query: str = ""  # pandas query, dotted names need `backticks`
    mask_fn: Callable[[pd.Series], bool] | None = field(default=None, cmd=False)
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = field(default=None, cmd=False)  # after filtering
    value: str = "best_eval_reward"  # score column, or history key for line plots
    by: list[str] = field(default_factory=list)  # boxes / bars / lines / table columns
    split_by: str = ""  # one subplot per value
    figure_by: str = ""  # one figure per value
    per_seed: Literal["best", "latest", "none"] = "best"  # dedup runs per (by, split_by, seed)
    seed: str = "seed"
    seed_keys: list[str] = field(default_factory=list)  # extra columns to dedup per seed within
    maximize: bool = True
    normalize: bool = False  # min-max normalize value per normalize_by (default split_by)
    normalize_by: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)  # preferred order of by labels
    split_order: list[str] = field(default_factory=list)

    # Plot options
    agg: str = "mean"
    shade: str = "std"  # std, var, sem, minmax or "" (line plots)
    err: str = "std"  # error bars (bar plots)
    shade_alpha: float = 0.3  # band opacity (line plots)
    window: int = 0  # rolling mean (line plots)
    downsample: int = 1
    x: str = "step"  # x column (line, scatter)
    x_scale: float = 1.0
    run_col: str = "run_id"
    events: list[float] = field(default_factory=list)  # vertical markers (line plots)
    color_by: str = ""  # scatter
    marker_by: str = ""  # scatter
    colors: dict = field(default_factory=dict, cmd=False)
    linestyles: dict = field(default_factory=dict, cmd=False)
    markers: dict = field(default_factory=dict, cmd=False)
    label_fn: Callable[[pd.Series], str] | None = field(default=None, cmd=False)  # box tick labels
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    legend_title: str = ""
    legend_loc: str = "best"
    xlim: list[float] = field(default_factory=list)
    ylim: list[float] = field(default_factory=list)
    logy: bool = False
    figsize: list[float] = field(default_factory=list)
    rows: int = 1
    sharey: bool = False
    rot: float = 30

    # Table options
    index: str = ""  # table rows
    measure: str = "mean"
    decimals: int = 2
    table_format: Literal["latex", "md"] = "latex"
    row_order: list[str] = field(default_factory=list)
    col_order: list[str] = field(default_factory=list)
    transpose: bool = False

    # Output
    out: str = ""  # file stem, default name
    out_dir: str = P.PLOT_DIR
    formats: list[str] = field(default_factory=lambda: ["pdf"])
    transparent: bool = False
    show: bool = False

    def replace(self, **kw) -> "EvalConfig":
        """Copy with some fields changed."""
        return dataclasses.replace(self, **kw)


# --------------------------------------------------------------------------- #
def load(cfg: EvalConfig, derive: Callable | None = None) -> pd.DataFrame:
    """Load the data a config asks for, downloading from wandb if needed.

    Parameters
    ----------
    cfg : EvalConfig
        Config with ``loader``, ``data`` or ``project`` set.
    derive : Callable, optional
        For histories, applied to the run metadata before the join.

    Returns
    -------
    pd.DataFrame
        Run table, or long history table for line plots.
    """
    if cfg.loader is not None:
        return cfg.loader(cfg)
    history = cfg.kind == "line"
    paths = list(cfg.data)
    if not paths:
        if not cfg.project:
            raise ValueError("Set data, project or loader")
        paths = D.download_runs(
            cfg.project, cfg.sweeps or None, cfg.wandb_filters, cfg.entity, cfg.data_dir, cfg.download,
            cfg.history_keys or ([cfg.value] if history else None), cfg.x_key,
        )
        if history:
            paths =[D.cache_path(cfg.project, s, True, cfg.data_dir) for s in (cfg.sweeps or [None])]
    return D.load_history(paths, meta_fn=derive) if history else D.load_runs(paths)


def prepare(cfg: EvalConfig, df: pd.DataFrame, derive: Callable | None = None) -> pd.DataFrame:
    """Derive columns, filter, deduplicate and normalize.

    Parameters
    ----------
    cfg : EvalConfig
        Selection settings.
    df : pd.DataFrame
        Loaded data.
    derive : Callable, optional
        Repo hook ``df -> df`` that adds computed columns.

    Returns
    -------
    pd.DataFrame
        Data ready for plotting.
    """
    if derive is not None:
        df = derive(df)
    if cfg.fields:
        df = D.pull_fields(df, cfg.fields)
    if cfg.rename:
        df = S.rename_values(df, cfg.rename)
    value = cfg.value
    if cfg.kind == "line" and "key" in df.columns:
        keys = set(df["key"].unique())
        if value not in keys and len(keys) > 1:
            raise KeyError(f"History key {value!r} not found, available: {sorted(keys)}")
        df = df[df["key"] == value] if value in keys else df
        value = "value"
    if cfg.kind not in ("line", "scatter"):
        df = S.to_numeric(df, value)
    df = S.filter_runs(df, cfg.query, cfg.mask_fn)
    if value in df.columns:
        df = df.dropna(subset=[value])
    if cfg.transform is not None:
        df = cfg.transform(df)
    if cfg.kind not in ("line", "scatter", "download") and cfg.per_seed != "none":
        keys = cfg.by + [k for k in (cfg.split_by, cfg.figure_by, cfg.index) if k] + cfg.seed_keys
        if cfg.per_seed == "best":
            df = S.best_per_seed(df, value, keys, cfg.seed, cfg.maximize)
        else:
            df = S.latest_per_seed(df, keys, cfg.seed)
    if cfg.normalize:
        df = S.normalize_per(df, value, cfg.normalize_by or [cfg.split_by])
    return df


def _finish_kw(cfg):
    return dict(
        title=cfg.title, ylim=cfg.ylim or None, xlim=cfg.xlim or None, ylabel=cfg.ylabel, xlabel=cfg.xlabel,
        logy=cfg.logy,
    )


def make_output(cfg: EvalConfig, df: pd.DataFrame, suffix: str = ""):
    """Draw the plot or table for prepared data and save it.

    Parameters
    ----------
    cfg : EvalConfig
        Plot settings.
    df : pd.DataFrame
        Output of ``prepare``.
    suffix : str, optional
        Appended to the file stem (used by ``figure_by``).

    Returns
    -------
    matplotlib.figure.Figure or tuple or dict
        Figure, ``(table, text)`` for tables, or the summary dict.
    """
    value = "value" if cfg.kind == "line" and "key" in df.columns else cfg.value
    figsize = tuple(cfg.figsize) if cfg.figsize else None
    common = dict(figsize=figsize, **_finish_kw(cfg))
    if cfg.kind == "box":
        print(df.groupby([k for k in [cfg.split_by, *cfg.by] if k], dropna=False)[value].count().to_string())
        fig = P.boxplot(df, value, cfg.by, cfg.split_by, cfg.order, cfg.split_order, cfg.label_fn, cfg.rows,
                        sharey=cfg.sharey, rot=cfg.rot, **common)
    elif cfg.kind == "bar":
        fig = P.barplot(df, value, cfg.by, cfg.split_by, cfg.agg, cfg.err, cfg.order, cfg.split_order,
                        cfg.colors or None, cfg.rows, sharey=cfg.sharey, rot=cfg.rot, **common)
    elif cfg.kind == "line":
        fig = P.lineplot(df, cfg.by, cfg.x, value, cfg.run_col, cfg.split_by, cfg.agg, cfg.shade, cfg.window,
                         cfg.downsample, cfg.x_scale, cfg.events, cfg.order, cfg.split_order, cfg.colors or None,
                         cfg.linestyles or None, cfg.legend_title, cfg.legend_loc, cfg.shade_alpha, rows=cfg.rows,
                         sharey=cfg.sharey,
                         **common)
    elif cfg.kind == "scatter":
        fig = P.scatter(df, cfg.x, value, cfg.color_by, cfg.marker_by, cfg.markers or None,
                        legend_title=cfg.legend_title, **common)
    elif cfg.kind == "table":
        return T.make_table(df, value, cfg.index, cfg.by, cfg.measure, cfg.decimals, cfg.table_format,
                            cfg.row_order or None, cfg.col_order or None, cfg.transpose)
    elif cfg.kind == "summary":
        return T.summarize(df, value, cfg.by)
    else:
        raise ValueError(f"Unknown kind {cfg.kind}")
    P.save_fig(fig, (cfg.out or cfg.name or cfg.kind) + suffix, cfg.out_dir, cfg.formats, cfg.transparent)
    return fig


def run(cfg: EvalConfig, derive: Callable | None = None):
    """Load, prepare and plot (or tabulate) one config.

    Parameters
    ----------
    cfg : EvalConfig
        The preset.
    derive : Callable, optional
        Repo hook ``df -> df`` that adds computed columns.

    Returns
    -------
    object or list
        Result of ``make_output``; a list if ``figure_by`` is set; the cache
        paths for ``kind="download"``.
    """
    if cfg.kind == "download":
        return D.download_runs(
            cfg.project, cfg.sweeps or None, cfg.wandb_filters, cfg.entity, cfg.data_dir, cfg.download,
            cfg.history_keys or None, cfg.x_key,
        )
    history = cfg.kind == "line" and cfg.loader is None
    df = prepare(cfg, load(cfg, derive if history else None), None if history else derive)
    if df.empty:
        raise ValueError("No rows left after filtering")
    if cfg.figure_by:
        out = [
            make_output(cfg.replace(title=cfg.title or str(k).split("/")[-1]), d, "_" + str(k).split("/")[-1])
            for k, d in df.groupby(cfg.figure_by, sort=False)
        ]
    else:
        out = make_output(cfg, df)
    if cfg.show:
        plt.show()
    return out


def main(
    presets: dict[str, EvalConfig] | None = None,
    derive: Callable | None = None,
    commands: dict[str, Callable[[list[str]], object]] | None = None,
    argv: list[str] | None = None,
):
    """Command line entry point: ``<preset|command> [--field value ...]``.

    Parameters
    ----------
    presets : dict[str, EvalConfig], optional
        Named presets. Any ``EvalConfig`` field can be overridden on the
        command line. Without presets the whole config comes from the CLI.
    derive : Callable, optional
        Repo hook passed to ``run``.
    commands : dict[str, Callable], optional
        Extra commands that receive the remaining arguments (e.g. ``fix_runs``).
    argv : list[str], optional
        Arguments, by default ``sys.argv[1:]``.

    Returns
    -------
    object
        Whatever ``run`` or the command returns.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    presets, commands = presets or {}, commands or {}
    if argv and argv[0] in ("-l", "--list") or (presets and not argv):
        print("Presets:")
        for k, v in presets.items():
            print(f"  {k:30s} {v.kind:8s} {v.title}")
        for k in commands:
            print(f"  {k:30s} command")
        return None
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    default = None
    if argv and not argv[0].startswith("-"):
        name = argv.pop(0)
        if name not in presets:
            raise SystemExit(f"Unknown preset {name!r}, use --list")
        default = presets[name].replace(name=presets[name].name or name)
    cfg = simple_parsing.parse(EvalConfig, args=argv, default=default)
    if default is not None:  # simple_parsing drops cmd=False fields, restore them
        for f in dataclasses.fields(EvalConfig):
            if not f.metadata.get("cmd", True):
                setattr(cfg, f.name, getattr(default, f.name))
    return run(cfg, derive)
