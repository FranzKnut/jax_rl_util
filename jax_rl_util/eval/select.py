"""Filter, deduplicate and normalize run tables."""

from collections.abc import Callable

import numpy as np
import pandas as pd


def _as_list(x) -> list:
    if x is None or x == "":
        return []
    return [x] if isinstance(x, str) else list(x)


def filter_runs(
    df: pd.DataFrame,
    query: str = "",
    mask_fn: Callable[[pd.Series], bool] | None = None,
    isin: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Keep the rows that match all given conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    query : str, optional
        ``DataFrame.query`` expression. Dotted columns need backticks,
        e.g. ``"`config.env_params.obs_mask` == 'even'"``.
    mask_fn : Callable, optional
        Row-wise predicate, by default None.
    isin : dict[str, list], optional
        Column -> allowed values, by default None.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    if query:
        df = df.query(query, engine="python")
    for col, values in (isin or {}).items():
        df = df[df[col].isin(values)]
    if mask_fn is not None and len(df):
        df = df[df.apply(mask_fn, axis=1).astype(bool)]
    return df.copy()


def best_per_seed(
    df: pd.DataFrame, value: str, by, seed: str = "seed", maximize: bool = True
) -> pd.DataFrame:
    """Keep the best run for each group and seed.

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    value : str
        Score column.
    by : str or list[str]
        Grouping columns (model, env, ...).
    seed : str, optional
        Seed column, by default "seed". Skipped if missing.
    maximize : bool, optional
        Keep the highest value if True, the lowest otherwise.

    Returns
    -------
    pd.DataFrame
        One row per (by, seed).
    """
    keys = _as_list(by) + ([seed] if seed in df.columns else [])
    if not keys:
        return df
    df = df.sort_values(value, ascending=maximize, na_position="first")
    return df.groupby(keys, dropna=False, sort=False).tail(1)


def latest_per_seed(df: pd.DataFrame, by, seed: str = "seed", time_col: str = "created_at") -> pd.DataFrame:
    """Keep the most recent run for each group and seed.

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    by : str or list[str]
        Grouping columns.
    seed : str, optional
        Seed column, by default "seed".
    time_col : str, optional
        Creation time column, by default "created_at".

    Returns
    -------
    pd.DataFrame
        One row per (by, seed).
    """
    keys = _as_list(by) + ([seed] if seed in df.columns else [])
    df = df.sort_values(time_col)
    return df.groupby(keys, dropna=False, sort=False).tail(1)


def normalize_per(df: pd.DataFrame, value: str, by, out: str | None = None) -> pd.DataFrame:
    """Min-max normalize ``value`` within each group.

    The normalized score is

    .. math:: \\tilde v = \\frac{v - \\min_g v}{\\max_g v - \\min_g v}

    where :math:`g` is the group (usually the environment).

    Parameters
    ----------
    df : pd.DataFrame
        Run table.
    value : str
        Column to normalize.
    by : str or list[str]
        Groups to normalize within.
    out : str, optional
        Output column, by default overwrites ``value``.

    Returns
    -------
    pd.DataFrame
        Copy with the normalized column. Constant groups map to 1.
    """
    df = df.copy()

    def _mm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo) if hi > lo else x * 0 + 1.0

    df[out or value] = df.groupby(_as_list(by), dropna=False)[value].transform(_mm)
    return df


def rename_values(df: pd.DataFrame, mapping: dict[str, dict]) -> pd.DataFrame:
    """Replace values per column, e.g. ``{"model_name": {"bptt": "CT-RNN"}}``.

    Parameters
    ----------
    df : pd.DataFrame
        Table.
    mapping : dict[str, dict]
        Column -> {old: new}. Columns that do not exist are skipped.

    Returns
    -------
    pd.DataFrame
        Copy with replaced values.
    """
    df = df.copy()
    for col, m in mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(m)
    return df


def to_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Coerce columns to float, mapping +-inf and unparsable values to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Table.
    cols : str or list[str]
        Columns to convert.

    Returns
    -------
    pd.DataFrame
        Copy with numeric columns.
    """
    df = df.copy()
    for c in _as_list(cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def group_label(df: pd.DataFrame, by, sep: str = " ") -> pd.Series:
    """Join the ``by`` columns into one string label per row.

    Parameters
    ----------
    df : pd.DataFrame
        Table.
    by : str or list[str]
        Columns to join.
    sep : str, optional
        Separator, by default " ".

    Returns
    -------
    pd.Series
        Labels; empty strings if ``by`` is empty.
    """
    by = _as_list(by)
    if not by:
        return pd.Series([""] * len(df), index=df.index)
    if len(by) == 1:
        return df[by[0]].astype(str)
    return df[by].astype(str).agg(sep.join, axis=1).str.strip()


def ordered(values, order: list | None = None) -> list:
    """Sort unique values, putting those listed in ``order`` first in that order.

    Parameters
    ----------
    values : iterable
        Values to order.
    order : list, optional
        Preferred order. Values not listed follow, sorted.

    Returns
    -------
    list
        Ordered unique values.
    """
    uniq = list(dict.fromkeys(values))
    if not order:
        try:
            return sorted(uniq)
        except TypeError:
            return sorted(uniq, key=str)
    first = [o for o in order if o in uniq]
    rest = [u for u in uniq if u not in first]
    try:
        rest = sorted(rest)
    except TypeError:
        rest = sorted(rest, key=str)
    return first + rest
