"""Aggregate runs into LaTeX or Markdown tables."""

from typing import Literal

import pandas as pd

from jax_rl_util.eval.select import _as_list


def _to_markdown(table: pd.DataFrame) -> str:
    """Markdown table, without needing ``tabulate``."""
    try:
        return table.to_markdown()
    except ImportError:
        def _name(c):
            return " ".join(map(str, c)) if isinstance(c, tuple) else str(c)

        head = [_name(table.index.name or "")] + [_name(c) for c in table.columns]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for idx, row in table.iterrows():
            lines.append("| " + " | ".join([_name(idx)] + [str(v) for v in row]) + " |")
        return "\n".join(lines)


def _to_latex(table: pd.DataFrame) -> str:
    """LaTeX tabular, falling back to a plain writer if ``jinja2`` is missing."""
    try:
        return table.to_latex(escape=False, column_format="l" + "c" * table.shape[1], multicolumn_format="c")
    except ImportError:
        def _name(c):
            return " ".join(map(str, c)) if isinstance(c, tuple) else str(c)

        lines = [
            "\\begin{tabular}{l" + "c" * table.shape[1] + "}",
            "\\toprule",
            " & ".join([_name(table.index.name or "")] + [_name(c) for c in table.columns]) + " \\\\",
            "\\midrule",
        ]
        lines += [" & ".join([_name(i)] + [str(v) for v in row]) + " \\\\" for i, row in table.iterrows()]
        lines += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)


def make_table(
    df: pd.DataFrame,
    value: str,
    index: str,
    columns,
    measure: str = "mean",
    decimals: int = 2,
    fmt: Literal["latex", "md"] = "latex",
    row_order: list | None = None,
    col_order: list | None = None,
    transpose: bool = False,
    show_count: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Make a ``measure ± std`` table from a run table.

    Parameters
    ----------
    df : pd.DataFrame
        Run table, already filtered and deduplicated per seed.
    value : str
        Score column.
    index : str
        Column for the table rows.
    columns : str or list[str]
        Column(s) for the table columns.
    measure : str, optional
        Central measure, e.g. "mean" or "median", by default "mean".
    decimals : int, optional
        Decimal places, by default 2.
    fmt : {"latex", "md"}, optional
        Output format, by default "latex".
    row_order, col_order : list, optional
        Reindex rows / columns to this order (missing entries become empty).
    transpose : bool, optional
        Swap rows and columns, by default False.
    show_count : bool, optional
        Append ``(n)`` with the number of seeds to each cell, by default False.
    verbose : bool, optional
        Print the raw pivot and the formatted table, by default True.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Formatted table and its LaTeX/Markdown string.
    """
    columns = _as_list(columns)
    if df.empty:
        raise ValueError("No rows to tabulate")
    pivot = df.pivot_table(
        index=index, columns=columns, values=value, aggfunc=[measure, "std", "count"], sort=False
    )
    if verbose:
        print(pivot, "\n")
    f = f"{{:.{decimals}f}}"
    pm = " $\\pm$ " if fmt == "latex" else " ± "
    center, std, count = pivot[measure], pivot["std"], pivot["count"]
    # A single seed has no std; show the value alone
    table = center.map(f.format) + (pm + std.fillna(0).map(f.format)).where(std.notna(), "")
    if show_count:
        table = table + " (" + count.astype(int).astype(str) + ")"
    table = table.where(center.notna(), "")
    if row_order is not None:
        table = table.reindex(row_order)
    if col_order is not None:
        table = table.reindex(col_order, axis=1)
    table = table.fillna("")
    if transpose:
        table = table.T
    if fmt == "md":
        out = _to_markdown(table)
    else:
        out = _to_latex(table)
    if verbose:
        print(out)
    return table, out


def summarize(df: pd.DataFrame, value: str, by, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Compare each hyperparameter on its own.

    For every column :math:`h` in ``by`` the runs are grouped by the value of
    :math:`h` alone and the (usually per-env normalized) score is averaged.

    Parameters
    ----------
    df : pd.DataFrame
        Run table, typically after ``normalize_per``.
    value : str
        Score column.
    by : str or list[str]
        Hyperparameter columns.
    verbose : bool, optional
        Print the tables, by default True.

    Returns
    -------
    dict[str, pd.DataFrame]
        Column -> table with ``mean``, ``std``, ``median``, ``count``, sorted by mean.
    """
    out = {}
    for col in _as_list(by):
        t = (
            df.groupby(col, dropna=False)[value]
            .agg(["mean", "std", "median", "count"])
            .sort_values("mean", ascending=False)
        )
        out[col] = t
        if verbose:
            print(f"\n== {col}\n{t.to_string(float_format='{:.3f}'.format)}")
    return out
