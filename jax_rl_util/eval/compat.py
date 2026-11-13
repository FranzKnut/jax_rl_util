"""Old function names, kept so existing notebooks keep working.

New code should use ``data.load_runs`` / ``data.pull_fields``,
``tables.make_table`` and ``data.download_runs``.
"""

import pandas as pd

from jax_rl_util.eval import data as D
from jax_rl_util.eval import select as S
from jax_rl_util.eval import tables as T


def gen_dict_extract(var, key):
    """Yield all values of ``key`` in a nested dict, breadth first."""
    if hasattr(var, "items"):
        if key in var:
            yield var[key]
        for v in var.values():
            if isinstance(v, dict):
                yield from gen_dict_extract(v, key)
            elif isinstance(v, list):
                for d in v:
                    yield from gen_dict_extract(d, key)


def deep_get(dictionary, key, default=None):
    """First value of ``key`` anywhere in a nested dict."""
    return next(gen_dict_extract(dictionary, key), default)


def pull_fields(df: pd.DataFrame, names: list[str] = ()) -> pd.DataFrame:
    """Old ``eval_util.pull_fields``: copy config fields found with ``deep_get`` into columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a ``config`` column (dict or string).
    names : list[str], optional
        Leaf names to pull.

    Returns
    -------
    pd.DataFrame
        Copy with the parsed config and one column per name.
    """
    df = df.copy()
    df["config"] = df["config"].apply(D.parse_config)
    for n in names:
        df[n] = df["config"].apply(lambda c, n=n: D._hashable(deep_get(c, n)))
    return df


def make_table(
    csv_path: str,
    fields: list[str],
    val_field: str,
    by: list[str],
    index: str,
    measure: str = "median",
    sweeps: list[str] | None = None,
    mask_fn=None,
    sort_by_value: bool = True,
    print_table: bool = True,
    decimal_places: int = 2,
    output_format: str = "latex",
):
    """Old ``make_latex_table.make_table`` signature on top of ``tables.make_table``.

    Column names have ``_`` replaced by spaces, as before.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Table and its LaTeX/Markdown string.
    """
    df = D.load_runs(csv_path, flatten=False)
    if sweeps is not None:
        df = df[df["Sweep"].isin(sweeps)]
    df = pull_fields(df, fields)
    df[fields] = df[fields].fillna("none")
    df.columns = [c.replace("_", " ") if isinstance(c, str) else c for c in df.columns]
    df = df.dropna(subset=[val_field])
    if mask_fn is not None:
        df = df[df.apply(mask_fn, axis=1)]
    if sort_by_value:
        df = S.best_per_seed(df, val_field, by + [index])
    else:
        df = S.latest_per_seed(df, by + [index], time_col="created at")
    return T.make_table(df, val_field, index, by, measure, decimal_places, output_format, verbose=print_table)


def wandb_to_csv(projects: list[str], sweeps: list[str] | None = None, out_dir: str = D.DATA_DIR, force: bool = False):
    """Old name of ``data.download_runs`` for several projects."""
    return [p for proj in projects for p in D.download_runs(proj, sweeps, out_dir=out_dir, force=force)]
