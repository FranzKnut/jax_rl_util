"""Download, cache and load experiment results.

All loaders return a ``pandas.DataFrame``. Run tables have one row per run with
a ``config`` column holding the run config as a dict. ``flatten_config`` turns
that into dotted columns such as ``config.env_params.env_name``. History tables
are in long format with one row per (run, key, step).
"""

import ast
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ENTITY = "franzknut"
DATA_DIR = "data/eval"
CONFIG_PREFIX = "config."

_TUPLE_COL = re.compile(r"^\((['\"])(.*)\1,\s*(-?\d+)\)$")


# --------------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------------- #
def parse_config(cfg):
    """Turn a stored config into a dict.

    Parameters
    ----------
    cfg : dict or str or None
        A dict, a JSON string (new caches) or a Python repr string (old CSVs).

    Returns
    -------
    dict
        The parsed config, or an empty dict if ``cfg`` is missing.
    """
    if isinstance(cfg, dict):
        return cfg
    if cfg is None or (isinstance(cfg, float) and np.isnan(cfg)):
        return {}
    if isinstance(cfg, str):
        try:
            return json.loads(cfg)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(cfg)
        except (ValueError, SyntaxError):
            # Old reprs can contain inf/nan or other names, fall back to eval
            # with an empty namespace.
            return eval(cfg, {"__builtins__": {}}, {"inf": np.inf, "nan": np.nan, "array": np.array})  # noqa: S307
    raise TypeError(f"Cannot parse config of type {type(cfg)}")


def _hashable(v):
    """Make values usable as groupby keys."""
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return str(v)
    return v


def flatten_config(df: pd.DataFrame, column: str = "config", prefix: str = CONFIG_PREFIX) -> pd.DataFrame:
    """Add one dotted column per (nested) config entry.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a ``column`` of config dicts or strings.
    column : str, optional
        Name of the config column, by default "config".
    prefix : str, optional
        Prefix for the new columns, by default "config.".

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the parsed config in ``column`` and the flattened
        entries as ``<prefix><a>.<b>`` columns. Lists become tuples.
    """
    if column not in df.columns:
        return df
    df = df.copy()
    df[column] = df[column].apply(parse_config)
    flat = pd.json_normalize(df[column].tolist())
    flat = flat.map(_hashable)
    flat.index = df.index
    flat.columns = [prefix + c for c in flat.columns]
    flat = flat.drop(columns=[c for c in flat.columns if c in df.columns])
    return pd.concat([df, flat], axis=1)


def resolve_field(df: pd.DataFrame, name: str) -> str:
    """Find the column that holds config field ``name``.

    Parameters
    ----------
    df : pd.DataFrame
        Table with flattened config columns.
    name : str
        A column name, a dotted path relative to the config (``env_params.env_name``)
        or a leaf name (``env_name``).

    Returns
    -------
    str
        The matching column. Exact matches win, then ``config.<name>``, then the
        shortest column that ends with ``.<name>``, so top-level keys beat nested
        ones (like ``deep_get`` did).

    Raises
    ------
    KeyError
        If no column matches.
    """
    if name in df.columns:
        return name
    if CONFIG_PREFIX + name in df.columns:
        return CONFIG_PREFIX + name
    candidates = [c for c in df.columns if isinstance(c, str) and c.endswith("." + name)]
    if not candidates:
        raise KeyError(f"No column matches field {name!r}")
    return min(candidates, key=lambda c: (c.count("."), c))


def pull_fields(df: pd.DataFrame, fields) -> pd.DataFrame:
    """Copy config fields into plain columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table with flattened config columns (see ``flatten_config``).
    fields : list[str] or dict[str, str]
        Either names resolved with ``resolve_field`` (the column keeps the leaf
        name), entries of the form ``"alias=path"``, or a dict alias -> path.
        Missing fields become columns of NaN.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the new columns. Existing columns with the same name
        are overwritten.
    """
    if isinstance(fields, dict):
        items = list(fields.items())
    else:
        items = []
        for f in fields:
            alias, _, path = f.partition("=")
            items.append((alias, path or alias))
    df = df.copy()
    for alias, path in items:
        alias = alias.split(".")[-1] if alias == path else alias
        try:
            col = resolve_field(df, path)
        except KeyError:
            df[alias] = np.nan
            continue
        if col != alias:
            df[alias] = df[col]
    return df


# --------------------------------------------------------------------------- #
# Cache files
# --------------------------------------------------------------------------- #
def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


def _cache_ext(fmt: str) -> str:
    if fmt == "auto":
        fmt = "parquet" if _has_parquet() else "json"
    return {"parquet": ".parquet", "json": ".json"}[fmt]


def save_table(df: pd.DataFrame, path: str) -> str:
    """Save a run or history table. Dict columns are stored as JSON strings.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    path : str
        Target path ending in ``.parquet``, ``.json`` or ``.csv``.

    Returns
    -------
    str
        The path written.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    if path.endswith(".parquet"):
        out.to_parquet(path, index=False)
    elif path.endswith(".json"):
        out.to_json(path, orient="records", lines=True, date_format="iso", default_handler=str)
    else:
        out.to_csv(path, index=False)
    return path


def _read_one(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".json"):
        return pd.read_json(path, orient="records", lines=True)
    df = pd.read_csv(path, low_memory=False)
    # Old caches were written with the index as first unnamed column
    first = df.columns[0]
    if isinstance(first, str) and (first == "" or first.startswith("Unnamed")):
        df = df.drop(columns=first)
    return df


def _expand(source) -> list[str]:
    if isinstance(source, str):
        source = [source]
    paths = []
    for s in source:
        if os.path.isdir(s):
            paths += sorted(
                p
                for p in glob.glob(os.path.join(s, "*"))
                if p.endswith((".csv", ".json", ".parquet")) and "_history" not in os.path.basename(p)
            )
        elif any(ch in s for ch in "*?["):
            paths += sorted(glob.glob(s))
        else:
            paths.append(s)
    return paths


def cache_path(project: str, sweep: str | None = None, history: bool = False, out_dir: str = DATA_DIR, fmt="auto"):
    """Path of the cache file for one sweep (or a whole project).

    Parameters
    ----------
    project : str
        Project name, with or without entity.
    sweep : str, optional
        Sweep id, by default the whole project ("all").
    history : bool, optional
        Whether this is the history cache, by default False.
    out_dir : str, optional
        Base directory, by default "data/eval".
    fmt : str, optional
        "parquet", "json" or "auto", by default "auto". An existing cache in any
        format is returned before a new name is made, including old
        ``<out_dir>/<sweep>.csv`` files.

    Returns
    -------
    str
        ``<out_dir>/<project>/<sweep>[_history].<ext>``
    """
    name = (sweep or "all") + ("_history" if history else "")
    stem = os.path.join(out_dir, project.split("/")[-1], name)
    for ext in (".parquet", ".json", ".csv"):
        if os.path.exists(stem + ext):
            return stem + ext
    # Old wandb_to_csv wrote <out_dir>/<sweep>.csv without the project folder
    legacy = os.path.join(out_dir, name + ".csv")
    if sweep is not None and os.path.exists(legacy):
        return legacy
    return stem + _cache_ext(fmt)


# --------------------------------------------------------------------------- #
# wandb download
# --------------------------------------------------------------------------- #
def _run_rows(runs, history_keys=None, x_key="_step", max_step=None):
    from tqdm import tqdm

    rows, hist = [], []
    for run in tqdm(runs):
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        rows.append(
            {
                "name": run.name,
                "run_id": run.id,
                "Sweep": run.sweep.id if run.sweep is not None else "none",
                "state": run.state,
                "created_at": pd.to_datetime(run.created_at),
                "config": config,
                **{k: v for k, v in run.summary._json_dict.items() if not isinstance(v, dict) or k == "_wandb"},
            }
        )
        for key in history_keys or []:
            kwargs = {"keys": [key, x_key], "page_size": 10_000}
            if max_step is not None:
                kwargs["max_step"] = int(max_step)
            for r in run.scan_history(**kwargs):
                hist.append({"run_id": run.id, "key": key, "step": r.get(x_key), "value": r.get(key)})
    runs_df = pd.DataFrame(rows)
    if "_wandb" in runs_df:
        runs_df = runs_df.drop(columns="_wandb")
    return runs_df, pd.DataFrame(hist, columns=["run_id", "key", "step", "value"])


def download_runs(
    project: str,
    sweeps: list[str] | None = None,
    filters: dict | None = None,
    entity: str = ENTITY,
    out_dir: str = DATA_DIR,
    force: bool = False,
    history_keys: list[str] | None = None,
    x_key: str = "_step",
    max_step: int | None = None,
    fmt: str = "auto",
) -> list[str]:
    """Download runs (and optionally their history) from wandb, one cache file per sweep.

    Parameters
    ----------
    project : str
        wandb project, ``"name"`` or ``"entity/name"``.
    sweeps : list[str], optional
        Sweep ids. By default all runs of the project go into ``all.<ext>``.
    filters : dict, optional
        Extra wandb API filters (MongoDB style), by default None.
    entity : str, optional
        Entity used if ``project`` has none, by default "franzknut".
    out_dir : str, optional
        Base cache directory, by default "data/eval".
    force : bool, optional
        Re-download even if the cache exists, by default False.
    history_keys : list[str], optional
        Metrics to fetch with ``scan_history``; stored in long format in
        ``<sweep>_history.<ext>``.
    x_key : str, optional
        History step key, by default "_step".
    max_step : int, optional
        Last step to fetch for histories, by default all.
    fmt : str, optional
        Cache format: "parquet", "json" or "auto" (parquet if pyarrow is
        installed), by default "auto".

    Returns
    -------
    list[str]
        Paths of the run caches.
    """
    api = None
    path = project if "/" in project else f"{entity}/{project}"
    paths = []
    for sweep in sweeps or [None]:
        out = cache_path(path, sweep, out_dir=out_dir, fmt=fmt)
        out_hist = cache_path(path, sweep, history=True, out_dir=out_dir, fmt=fmt)
        need_hist = bool(history_keys) and (force or not os.path.exists(out_hist))
        if os.path.exists(out) and not force and not need_hist:
            print(f"Using cached {out}")
            paths.append(out)
            continue
        if api is None:
            import wandb

            api = wandb.Api(timeout=60)
        _filters = dict(filters or {})
        if sweep is not None:
            _filters["sweep"] = sweep
        print(f"Downloading {path} {sweep or ''} ...")
        runs_df, hist_df = _run_rows(api.runs(path, filters=_filters, per_page=100), history_keys, x_key, max_step)
        if sweep is not None:
            try:
                runs_df["Sweep_name"] = api.sweep(f"{path}/{sweep}").name
            except Exception:  # noqa: BLE001
                pass
        # Always write the new format into the project folder; old CSVs stay untouched
        stem = os.path.join(out_dir, path.split("/")[-1], sweep or "all")
        out, out_hist = stem + _cache_ext(fmt), stem + "_history" + _cache_ext(fmt)
        print("Saved", save_table(runs_df, out))
        if history_keys:
            print("Saved", save_table(hist_df, out_hist))
        paths.append(out)
    return paths


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_runs(source, flatten: bool = True) -> pd.DataFrame:
    """Load one or more run tables.

    Parameters
    ----------
    source : str or list[str]
        Files (``.parquet``, ``.json``, ``.csv``), directories or glob patterns.
        Old CSVs whose ``config`` column is a Python repr are read as well.
    flatten : bool, optional
        Add dotted ``config.*`` columns, by default True.

    Returns
    -------
    pd.DataFrame
        All runs, with a ``source`` column naming the file each came from.
    """
    paths = _expand(source)
    if not paths:
        raise FileNotFoundError(f"No run tables found in {source}")
    dfs = []
    for p in paths:
        d = _read_one(p)
        d["source"] = os.path.splitext(os.path.basename(p))[0]
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    if flatten:
        df = flatten_config(df)
    return df


def history_to_long(df: pd.DataFrame, id_col: str | None = None, meta_only: bool = False):
    """Convert an old wide history CSV (``"('key', i)"`` columns) to long format.

    Parameters
    ----------
    df : pd.DataFrame
        Wide table with one row per run, metadata columns and string tuple
        columns ``('<key>', i)``. Steps come from ``('<key>_step', i)``,
        ``('_step', i)`` or ``('Step', i)``, whichever exists.
    id_col : str, optional
        Column identifying runs, by default ``run_id`` or ``name`` or the row number.
    meta_only : bool, optional
        Return ``(meta, long)`` separately instead of the merged table.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        Long table with the metadata columns plus ``key``, ``step``, ``value``,
        or the run metadata and the long values (keyed by ``id_col``).
    """
    tuple_cols = {}
    meta_cols = []
    for c in df.columns:
        m = _TUPLE_COL.match(str(c))
        if m:
            tuple_cols.setdefault(m.group(2), []).append((int(m.group(3)), c))
        else:
            meta_cols.append(c)
    df = df.reset_index(drop=True)
    if id_col is None:
        id_col = next((c for c in ("run_id", "name") if c in df.columns and df[c].is_unique), None)
    if id_col is None:
        df["run_id"] = np.arange(len(df))
        meta_cols.append("run_id")
        id_col = "run_id"
    step_keys = {"_step", "Step"}
    value_keys = [k for k in tuple_cols if k not in step_keys and not k.endswith("_step")]
    out = []
    for key in value_keys:
        step_key = next((s for s in (f"{key}_step", "_step", "Step") if s in tuple_cols), None)
        cols = [c for _, c in sorted(tuple_cols[key])]
        vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if step_key is not None:
            scols = [c for _, c in sorted(tuple_cols[step_key])][: len(cols)]
            steps = df[scols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        else:
            steps = np.tile(np.arange(len(cols), dtype=float), (len(df), 1))
        long = pd.DataFrame(
            {
                id_col: np.repeat(df[id_col].to_numpy(), len(cols)),
                "key": key,
                "step": steps.ravel(),
                "value": vals.ravel(),
            }
        ).dropna(subset=["value"])
        out.append(long)
    long = pd.concat(out, ignore_index=True)
    meta = df[meta_cols]
    return (meta, long) if meta_only else meta.merge(long, on=id_col)


def load_history(source, runs: pd.DataFrame | None = None, flatten: bool = True, meta_fn=None) -> pd.DataFrame:
    """Load history tables in long format, joined with run metadata.

    Parameters
    ----------
    source : str or list[str]
        History files. New caches are already long (``run_id, key, step, value``);
        old wide CSVs are converted with ``history_to_long``.
    runs : pd.DataFrame, optional
        Run table to join on ``run_id``. For new caches it is loaded from the
        sibling file without ``_history`` if not given.
    flatten : bool, optional
        Add dotted ``config.*`` columns, by default True.
    meta_fn : Callable, optional
        ``df -> df`` applied to the run metadata (one row per run) before the
        join, e.g. a repo ``derive`` hook. Much faster than on the long table.

    Returns
    -------
    pd.DataFrame
        One row per (run, key, step) with all run columns.
    """
    dfs = []
    for p in _expand(source):
        d = _read_one(p)
        if {"key", "step", "value"}.issubset(d.columns):
            meta = runs
            if meta is None:
                sibling = p.replace("_history", "")
                meta = _read_one(sibling) if os.path.exists(sibling) else pd.DataFrame({"run_id": d["run_id"].unique()})
            meta, long, id_col = meta.drop(columns=[c for c in ("source",) if c in meta.columns]), d, "run_id"
        else:
            meta, long = history_to_long(d, meta_only=True)
            id_col = next(c for c in ("run_id", "name") if c in long.columns)
        meta = meta.copy()
        meta["source"] = os.path.splitext(os.path.basename(p))[0]
        if flatten:
            meta = flatten_config(meta)
        if meta_fn is not None:
            meta = meta_fn(meta)
        dfs.append(meta.merge(long, on=id_col, how="right"))
    return pd.concat(dfs, ignore_index=True)


def load_csv_folder(
    files: dict[str, str],
    value: str = "value",
    label: str = "model_name",
    step: str = "step",
    drop_rows=("sum", "avg"),
    skip_first_run: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Load hand-made CSVs with one column per run and one row per step.

    Parameters
    ----------
    files : dict[str, str]
        Label -> CSV path. The first CSV column is the step (e.g. ``lap``).
    value : str, optional
        Name for the value column, by default "value".
    label : str, optional
        Name for the label column, by default "model_name".
    step : str, optional
        Name for the step column, by default "step".
    drop_rows : tuple[str, ...], optional
        Summary rows to drop, by default ("sum", "avg").
    skip_first_run : tuple[str, ...], optional
        Labels whose first run column is dropped (e.g. a failed recording).

    Returns
    -------
    pd.DataFrame
        Long table with columns ``label``, ``run``, ``step``, ``value``.
    """
    out = []
    for name, path in files.items():
        d = pd.read_csv(path, index_col=0)
        d = d.drop(index=[r for r in drop_rows if r in d.index])
        if name in skip_first_run:
            d = d.iloc[:, 1:]
        d.index = pd.to_numeric(d.index)
        long = d.reset_index(names=step).melt(id_vars=step, var_name="run", value_name=value).dropna(subset=[value])
        long[label] = name
        out.append(long)
    return pd.concat(out, ignore_index=True)


def load_wandb_export(path: str, group_name: str | None = None) -> pd.DataFrame:
    """Load a CSV exported from a wandb line plot panel.

    Parameters
    ----------
    path : str
        CSV with a step column and ``"<group> - <metric>"`` columns plus optional
        ``__MIN`` / ``__MAX`` variants. ``<group>`` may look like
        ``"policy_config.num_modules: 10"``.
    group_name : str, optional
        Name for the group column, by default the part before ``:`` in the
        group labels (e.g. ``policy_config.num_modules``), or "group".

    Returns
    -------
    pd.DataFrame
        Long table with columns ``step``, ``<group_name>``, ``metric``, ``value``,
        ``min``, ``max``.
    """
    df = pd.read_csv(path)
    step_col = df.columns[0]
    rows = []
    for c in df.columns[1:]:
        base, stat = c, "value"
        for suffix, s in (("__MIN", "min"), ("__MAX", "max")):
            if c.endswith(suffix):
                base, stat = c[: -len(suffix)], s
        group, _, metric = base.rpartition(" - ")
        if not group:
            group, metric = base, ""
        rows.append((c, group, metric, stat))
    keys = [g for _, g, _, _ in rows]
    if group_name is None:
        prefixes = {g.split(": ")[0] for g in keys if ": " in g}
        group_name = prefixes.pop() if len(prefixes) == 1 else "group"
    long = []
    for c, group, metric, stat in rows:
        g = group.split(": ", 1)[1] if ": " in group else group
        long.append(pd.DataFrame({"step": df[step_col], group_name: g, "metric": metric, "stat": stat, "v": df[c]}))
    long = pd.concat(long, ignore_index=True)
    out = long.pivot_table(index=["step", group_name, "metric"], columns="stat", values="v").reset_index()
    out.columns.name = None
    return out
