"""Shared evaluation tools: load runs, select, plot and tabulate.

Typical use from a repo's ``<package>/eval.py``::

    from jax_rl_util.eval import EvalConfig, main

    PRESETS = {"box": EvalConfig(kind="box", project="RTRRL_2", sweeps=["xprah6y7"],
                                 fields=["model=rnn_config.model_name", "env=env_name", "seed"],
                                 by=["model"], split_by="env")}

    if __name__ == "__main__":
        main(PRESETS, derive=None)
"""

from jax_rl_util.eval.data import (
    download_runs,
    flatten_config,
    history_to_long,
    load_csv_folder,
    load_history,
    load_runs,
    load_wandb_export,
    parse_config,
    pull_fields,
    resolve_field,
    save_table,
)
from jax_rl_util.eval.pipeline import EvalConfig, load, main, make_output, prepare, run
from jax_rl_util.eval.plots import barplot, boxplot, lineplot, save_fig, scatter
from jax_rl_util.eval.select import (
    best_per_seed,
    filter_runs,
    group_label,
    latest_per_seed,
    normalize_per,
    rename_values,
    to_numeric,
)
from jax_rl_util.eval.tables import make_table, summarize

__all__ = [
    "EvalConfig",
    "barplot",
    "best_per_seed",
    "boxplot",
    "download_runs",
    "filter_runs",
    "flatten_config",
    "group_label",
    "history_to_long",
    "latest_per_seed",
    "lineplot",
    "load",
    "load_csv_folder",
    "load_history",
    "load_runs",
    "load_wandb_export",
    "main",
    "make_output",
    "make_table",
    "normalize_per",
    "parse_config",
    "prepare",
    "pull_fields",
    "rename_values",
    "resolve_field",
    "run",
    "save_fig",
    "save_table",
    "scatter",
    "summarize",
    "to_numeric",
]
