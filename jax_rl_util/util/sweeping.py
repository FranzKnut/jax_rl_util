"""Utilities for creating wandb sweeps and reading their results.

A typical ``make_*_sweep.py`` script defines its sweep parameters, builds the
sweep config with `make_sweep_config` and hands it to `create_sweep`:

    sweep_config = make_sweep_config(PARAMS, MyConfig, program=__file__)
    create_sweep(sweep_config, MyConfig())

Every such script then accepts ``--dry-run``, ``--project`` and ``--entity``
on the command line (see `SweepConfig`). Scripts that run the trials
themselves check `is_trial_run` first.

wandb, pandas and tqdm are imported lazily, since they are not dependencies
of jax_rl_util itself.
"""

import dataclasses
import os
import sys
import time
import typing
from collections.abc import Callable, Mapping
from pprint import pprint

import simple_parsing
from simple_parsing import DashVariant

from jax_rl_util.util.logging_util import WANDB_ENTITY, LoggableConfig

TRIAL_FLAG = "--trial"


@dataclasses.dataclass
class SweepConfig:
    """Command line options shared by all sweep scripts."""

    project: str | None = None  # overrides the project given by the script
    entity: str | None = None  # overrides the entity given by the script
    dry_run: bool = False  # print the config without creating the sweep


# Building sweep configs


def make_sweep_config(
    params: dict,
    config_cls: type | None = None,
    program: str | os.PathLike | None = None,
    name: str | None = None,
    method: str = "grid",
    metric: str | None = None,
    goal: str = "maximize",
    command: list[str] | None = None,
) -> dict:
    """Build a wandb sweep config.

    Parameters
    ----------
    params : dict
        Sweep parameters.
    config_cls : type, optional
        Dataclass the runs decode their config into. If given, `params` are
        checked against its fields with `check_fields`.
    program : str or os.PathLike, optional
        Script the agent runs. Usually ``__file__``; it is stored relative to
        the current working directory, which should be the repo root that the
        agents are started from.
    name : str, optional
        Default sweep name.
    method : str, optional
        Search method: ``"grid"``, ``"random"`` or ``"bayes"``.
    metric : str, optional
        Name of the metric to optimize.
    goal : str, optional
        ``"maximize"`` or ``"minimize"`` `metric`.
    command : list[str], optional
        Command the agent runs. If `program` is given and `command` is not,
        the agent runs `program` with ``--trial``, see `is_trial_run`.

    Returns
    -------
    dict
        The wandb sweep config.

    Raises
    ------
    KeyError
        If `params` contain names that `config_cls` has no field for.
    """
    if config_cls is not None:
        check_fields(params, config_cls)
    sweep_config = {"method": method}
    if name:
        sweep_config["name"] = name
    if metric:
        sweep_config["metric"] = {"name": metric, "goal": goal}
    if program is not None:
        sweep_config["program"] = os.path.relpath(program)
        # No ${args}: runs read the sweep parameters from wandb.config in `with_logger`.
        command = command or ["${env}", "${interpreter}", "${program}", TRIAL_FLAG]
    if command is not None:
        sweep_config["command"] = command
    sweep_config["parameters"] = params
    return sweep_config


def update_sweep_dict(d: dict, u: Mapping) -> dict:
    """Update nested sweep dict `d` with values from nested dict `u`.

    Also makes sure that only one of the keys ``value`` and ``values`` is
    present, not both.

    Parameters
    ----------
    d : dict
        Base dict, updated in place.
    u : Mapping
        Updates.

    Returns
    -------
    dict
        `d` with values overwritten by `u`.
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_sweep_dict(d.get(k, {}), v)
        else:
            d[k] = v
        if k in ("value", "values"):
            other = "values" if k == "value" else "value"
            if other in d and len(d) == 2:
                print("Removing", other, "from", d)
                del d[other]
    return d


# Checking sweep configs


def _dataclass_in(tp) -> type | None:
    """Return the dataclass in type hint `tp`, e.g. for ``Config | None``."""
    if dataclasses.is_dataclass(tp):
        return tp
    for arg in typing.get_args(tp):
        if dataclasses.is_dataclass(arg):
            return arg
    return None


def _field_types(config_cls: type) -> dict:
    """Map the field names of `config_cls` to their resolved types."""
    try:
        hints = typing.get_type_hints(config_cls)
    except (NameError, TypeError):
        hints = {}
    return {f.name: hints.get(f.name, f.type) for f in dataclasses.fields(config_cls)}


def unknown_params(params: dict, config_cls: type, prefix: str = "") -> list[str]:
    """Find sweep parameters that are not fields of `config_cls`.

    Nested ``parameters`` blocks and dict-valued ``values`` are followed into
    nested config dataclasses, also through hints like ``Config | None``.
    Plain dict fields such as ``loss_weights`` take arbitrary keys and are
    not checked.

    Parameters
    ----------
    params : dict
        Sweep parameters, nested via ``{"parameters": {...}}``.
    config_cls : type
        Dataclass the runs decode their config into.
    prefix : str, optional
        Dotted path of `params` within the root config, for the messages.

    Returns
    -------
    list[str]
        Dotted names of the parameters that `config_cls` has no field for.
    """
    fields = _field_types(config_cls)
    unknown = []
    for key, spec in params.items():
        if key not in fields:
            unknown.append(prefix + key)
            continue
        sub_cls = _dataclass_in(fields[key])
        if sub_cls is None or not isinstance(spec, dict):
            continue
        nested = dict(spec.get("parameters", {}))
        for value in spec.get("values", []):
            if isinstance(value, dict):
                nested.update({k: {"value": v} for k, v in value.items()})
        if nested:
            unknown += unknown_params(nested, sub_cls, f"{prefix}{key}.")
    return unknown


def check_fields(params: dict, config_cls: type):
    """Check that every swept key names a field of `config_cls`.

    Parameters
    ----------
    params : dict
        Sweep parameters, see `unknown_params`.
    config_cls : type
        Dataclass the runs decode their config into.

    Raises
    ------
    KeyError
        If a key doesn't exist, which otherwise only shows up once the first
        agent has already loaded the data and started training.
    """
    unknown = unknown_params(params, config_cls)
    if unknown:
        raise KeyError(f"Not fields of {config_cls.__name__}: {', '.join(unknown)}")


# Describing sweep configs


def count_combinations(config) -> int:
    """Count the runs of a grid over a nested sweep config.

    Parameters
    ----------
    config : dict
        Sweep parameters, or a part of them.

    Returns
    -------
    int
        Product of the lengths of all ``values`` lists in `config`.
    """
    if not isinstance(config, dict):
        return 1
    if "values" in config:
        return len(config["values"])
    total = 1
    for value in config.values():
        total *= count_combinations(value)
    return total


def extract_keys_with_values(d: dict, parent_key: str = "") -> dict:
    """Collect the swept parameters of a nested sweep config.

    Parameters
    ----------
    d : dict
        Sweep parameters.
    parent_key : str, optional
        Dotted path of `d`, used as a prefix for the keys.

    Returns
    -------
    dict
        Maps dotted keys to their list of ``values``, or to a string such as
        ``"log_uniform_values [1e-05, 0.01]"`` for a distribution.
    """
    result = {}
    for k, v in d.items():
        key_path = f"{parent_key}.{k}" if parent_key else k
        if not isinstance(v, dict):
            continue
        if "values" in v:
            result[key_path] = v["values"]
        elif "distribution" in v:
            result[key_path] = (
                f"{v['distribution']} [{v.get('min', '')}, {v.get('max', '')}]"
            )
        else:
            result.update(extract_keys_with_values(v, key_path))
    return result


def _format_values(values) -> str:
    """Format an entry of `extract_keys_with_values` for printing."""
    if isinstance(values, str):
        return values
    return ", ".join(map(str, values))


def _is_swept(values) -> bool:
    """Whether an entry of `extract_keys_with_values` varies across runs."""
    return isinstance(values, str) or len(values) > 1


# Creating sweeps


def _git(repo_dir, *args) -> str:
    """Run a git command in `repo_dir` and return its stripped output."""
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True
    )
    return out.stdout.strip()


def create_sweep_interactively(
    sweep_config: dict,
    project: str | None = None,
    config_repo_dir=None,
    based_on_sweep: str | None = None,
    entity: str = WANDB_ENTITY,
    **kwargs,
) -> str:
    """Create a wandb sweep, asking for confirmation and a sweep name.

    A description of the sweep is written to ``sweeps/<name>.txt``.
    For more info on the config format, see the wandb documentation.

    Parameters
    ----------
    sweep_config : dict
        wandb sweep configuration.
    project : str, optional
        wandb project name.
    config_repo_dir : str or os.PathLike, optional
        Git repository that contains the code. If given, uncommitted changes
        are reported, the commit hash is logged and used as the default sweep
        name, and a ``sweep-<name>`` git tag is created.
    based_on_sweep : str, optional
        Sweep this one builds on, e.g. for pretrained models. Only logged.
    entity : str, optional
        wandb entity.
    **kwargs
        Passed on to `wandb.sweep`.

    Returns
    -------
    str
        Sweep path ``<entity>/<project>/<sweep_id>``.
    int
        Estimated number of runs in the sweep.
    """
    import wandb

    pprint(sweep_config)
    est_runs = count_combinations(sweep_config["parameters"])
    swept = extract_keys_with_values(sweep_config["parameters"])

    print("Entity:", entity)
    print("Project:", project)
    print("Est. runs:", est_runs)
    for k, v in swept.items():
        if _is_swept(v):
            print(f"- {k}: {_format_values(v)}")
    name = input(f'Enter custom sweep name ("{sweep_config.get("name", "")}"):  ')

    git_hash = None
    if config_repo_dir is not None:
        if _git(config_repo_dir, "status", "--porcelain"):
            print(
                f"Git repository at {os.path.abspath(config_repo_dir)} has uncommitted changes.\n"
                "Please commit or stash them before creating a sweep."
            )
            input("Press Enter to continue anyway, or Ctrl+C to abort.")
        git_hash = _git(config_repo_dir, "rev-parse", "--short", "HEAD")
        print(f"Git hash for sweep: {git_hash}")

    name = name or sweep_config.get("name") or git_hash or "sweep"
    sweep_config["name"] = name

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity, **kwargs)
    sweep_path = f"{entity}/{project}/{sweep_id}"

    os.makedirs("sweeps", exist_ok=True)
    lines = [
        "---",
        f"## Sweep {name} ({sweep_id})",
        "Created at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        f"Est. runs: {est_runs}",
        "",
        f"> ID:   {sweep_id}",
        "",
        "**Description**",
        *[f"- {k}: {_format_values(v)}" for k, v in swept.items()],
        "",
    ]
    if git_hash is not None:
        _git(config_repo_dir, "tag", f"sweep-{name.replace(' ', '-')}")
        lines.append(f"Git hash: {git_hash}")
    if based_on_sweep is not None:
        lines.append(f"Based on sweep: {based_on_sweep}")
    lines.append(f"URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    with open(f"sweeps/{name}.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    return sweep_path, est_runs


def parse_sweep_args(args: list[str] | None = None) -> SweepConfig:
    """Parse `SweepConfig` from the command line.

    Unknown arguments are ignored, so scripts can add their own.

    Parameters
    ----------
    args : list[str], optional
        Arguments to parse, by default ``sys.argv[1:]``.

    Returns
    -------
    SweepConfig
        The parsed options.
    """
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=DashVariant.UNDERSCORE_AND_DASH
    )
    parser.add_arguments(SweepConfig, "sweep")
    parsed, _ = parser.parse_known_args(args)
    return parsed.sweep


def create_sweep(
    sweep_config: dict,
    default: LoggableConfig | None = None,
    project: str | None = None,
    entity: str | None = None,
    config_repo_dir=".",
    based_on_sweep: str | None = None,
    print_agent_command: bool = True,
) -> str:
    """Print or create a sweep, as chosen by the `SweepConfig` command line options.

    The project and entity are taken from the first that is set of: the
    ``--project``/``--entity`` options, the `project`/`entity` arguments,
    ``default.project_name``/``default.repo``. The entity falls back to
    `WANDB_ENTITY`.

    Parameters
    ----------
    sweep_config : dict
        The wandb sweep config, e.g. from `make_sweep_config`.
    default : LoggableConfig, optional
        Default run config, for the project and entity.
    project : str, optional
        wandb project.
    entity : str, optional
        wandb entity.
    config_repo_dir : str or os.PathLike, optional
        See `create_sweep_interactively`. By default the working directory.
    based_on_sweep : str, optional
        See `create_sweep_interactively`.
    print_agent_command : bool, optional
        Whether to print how to start an agent for the new sweep.

    Returns
    -------
    str
        Sweep path ``<entity>/<project>/<sweep_id>``, or ``"dry-run"``.
    """
    options = parse_sweep_args()
    if options.dry_run:
        pprint(sweep_config)
        print("Est. runs:", count_combinations(sweep_config["parameters"]))
        return "dry-run"
    project = options.project or project or (default and default.project_name)
    entity = options.entity or entity or (default and default.repo) or WANDB_ENTITY
    sweep_path, est_runs = create_sweep_interactively(
        sweep_config,
        project,
        config_repo_dir=config_repo_dir,
        based_on_sweep=based_on_sweep,
        entity=entity,
    )
    if print_agent_command:
        print(f"Created sweep {sweep_path}. Start an agent with:")
        print(f'./launch_batch_slurm.sh "wandb agent {sweep_path}" {est_runs}')
    return sweep_path


def is_trial_run() -> bool:
    """Check for and remove the ``--trial`` flag that `make_sweep_config` adds.

    Returns
    -------
    bool
        Whether the script was started by a sweep agent to run one trial.
    """
    if TRIAL_FLAG in sys.argv:
        sys.argv.remove(TRIAL_FLAG)
        return True
    return False


# Reading sweep results


def _model_artifacts(run, artifact_filter: Callable | None = None) -> list:
    """Return the model artifacts logged by `run` that pass `artifact_filter`."""
    return [
        a
        for a in run.logged_artifacts()
        if a.type == "model" and (artifact_filter is None or artifact_filter(a))
    ]


def get_all_models_for_sweep(
    sweep_path: str,
    filter_fn: Callable | None = None,
    artifact_filter: Callable | None = None,
) -> list[str]:
    """Get the last model of every run in a sweep.

    Parameters
    ----------
    sweep_path : str
        Path to the sweep in the format ``"entity/project/sweep_id"``.
    filter_fn : Callable, optional
        Takes a run and returns True if it should be included.
    artifact_filter : Callable, optional
        Takes a model artifact and returns True if it may be picked; the run's
        last such artifact is used. By default every model artifact.

    Returns
    -------
    list[str]
        Model artifact paths.
    """
    import tqdm
    import wandb

    models = []
    for run in tqdm.tqdm(wandb.Api().sweep(sweep_path).runs):
        artifacts = _model_artifacts(run, artifact_filter)
        if not artifacts:
            print(f"Run {run.name} has no model artifacts, skipping.")
            continue
        if filter_fn and not filter_fn(run):
            continue
        models.append(artifacts[-1].source_qualified_name)
    return models


def get_representative_models_for_sweep(
    sweep_path: str,
    select: str = "best",
    group_keys: str | list[str] = "model_name",
    metric_key: str = "eval_reward",
    filter_fn: Callable | None = None,
    artifact_filter: Callable | None = None,
    minimize: bool = False,
) -> list[str]:
    """Get one representative model per group of runs in a sweep.

    Parameters
    ----------
    sweep_path : str
        Path to the sweep in the format ``"entity/project/sweep_id"``.
    select : str, optional
        How to select the representative of each group: ``"best"``,
        ``"worst"`` or ``"median"``.
    group_keys : str or list[str], optional
        Key(s) to group by, looked up in the run's top-level config first,
        then in its ``policy_config``.
    metric_key : str, optional
        Summary metric used for selecting the representative.
    filter_fn : Callable, optional
        Takes a run and returns True if it should be included.
    artifact_filter : Callable, optional
        Takes a model artifact and returns True if it may be picked; the run's
        last such artifact is used. By default every model artifact.
    minimize : bool, optional
        Whether a lower `metric_key` is better, e.g. for a loss.

    Returns
    -------
    list[str]
        Model artifact paths, one per group.
    """
    import pandas as pd
    import tqdm
    import wandb

    if select not in ("best", "worst", "median"):
        raise ValueError(f"Unknown select: {select}")
    if isinstance(group_keys, str):
        group_keys = [group_keys]

    rows = []
    for run in tqdm.tqdm(wandb.Api().sweep(sweep_path).runs):
        artifacts = _model_artifacts(run, artifact_filter)
        if not artifacts:
            print(f"Run {run.name} has no model artifacts, skipping.")
            continue
        if filter_fn and not filter_fn(run):
            continue
        cfg = run.load_full_data()["config"]
        group = tuple(
            cfg[key] if key in cfg else cfg.get("policy_config", {}).get(key)
            for key in group_keys
        )
        rows.append(
            {
                "group": group,
                metric_key: run.summary[metric_key],
                "model_path": artifacts[-1].source_qualified_name,
            }
        )
    if not rows:
        return []

    df = pd.DataFrame(rows)
    paths = []
    for _, runs in df.groupby("group", sort=False, dropna=False):
        metric = runs[metric_key]
        if select == "median":
            idx = metric.sub(metric.median()).abs().idxmin()
        elif (select == "best") != minimize:
            idx = metric.idxmax()
        else:
            idx = metric.idxmin()
        paths.append(runs.loc[idx, "model_path"])
    return paths
