"""Gather csv files and make latex table."""

from typing import Literal

import pandas as pd
from jax_rl_util.eval.eval_util import pull_fields


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
    output_format: Literal["latex", "md"] = "latex",
) -> tuple[pd.DataFrame, str]:
    """
    Create a LaTeX table from a CSV file with evaluation results.
    
    Args:
        csv_path: Path to the input CSV file
        fields: List of field names to extract and process
        val_field: Name of the value field to aggregate
        by: List of column names to group by (becomes table columns)
        index: Column name to use as table index (rows)
        measure: Aggregation measure (e.g., "median", "mean")
        sweeps: Optional list of sweep IDs to filter
        mask_fn: Optional filtering function applied to each row
        sort_by_value: If True, sort by val_field; if False, sort by "created at"
        print_table: If True, print intermediate and final tables
        decimal_places: Number of decimal places for formatting
        
    Returns:
        Tuple of (pivot_table_df, latex_string)
    """
    df = pd.read_csv(csv_path, index_col=0)
    
    # Filter by sweeps if provided
    if sweeps is not None:
        df = df[df["Sweep"].isin(sweeps)]
    
    # COLUMNS PRESENT IN DF ARE OVERWRITTEN!
    df = pull_fields(df, fields)
    df[fields] = df[fields].fillna("none")
    df[fields] = df[fields].apply(lambda r: r.str.replace("_", " ") if r.dtype == "object" else r)
    df.columns = df.columns.str.replace("_", " ")
    
    df = df.dropna(subset=[val_field])
    
    # Apply filtering function if provided
    if mask_fn is not None:
        df = df[df.apply(mask_fn, axis=1)]
    
    # Get the most recent or best run for each seed
    if sort_by_value:
        df = df.sort_values(val_field).groupby(by + [index, "seed"]).tail(1)
    else:
        df = df.sort_values("created at").groupby(by + [index, "seed"]).tail(1)
    
    def mean_pm_std(x):
        """Convert mean and std to latex string."""
        float_format = f"{{:.{decimal_places}f}}"
        return x[measure].map(lambda a: float_format.format(a)) + " $\pm$ " + x["std"].map(lambda a: float_format.format(a))
    
    # Make table
    pivot_df = df.pivot_table(
        index=index, columns=by, values=val_field, aggfunc={val_field: [measure, "std", "count"]}, sort=False
    )
    
    if print_table:
        print(pivot_df)
        print("")
        print("")
    
    result_df = pivot_df.apply(mean_pm_std, axis=1)
    result_df = result_df.transpose().sort_index(axis=1)
    
    if output_format == "md":
        latex_str = result_df.to_markdown()
    else:
        latex_str = result_df.to_latex(
            escape=False,
            column_format="l" + "c" * (result_df.shape[-1]),
            multicolumn_format="c",
            float_format=f"{{:.{decimal_places}f}}".format,
    )
    
    if print_table:
        print(latex_str)
    
    return result_df, latex_str


# Example usage
if __name__ == "__main__":
    FIELDS = ["env_name", "agent_type", "learning_rate", "seed", "obs_mask"]
    VAL_FIELD = "best eval"
    BY = ["agent type"]  # , "obs mask"
    INDEX = "env name"
    MEASURE = "median"
    
    # SWEEPS = ["olx8u5gy", "bkngzbt9"]
    
    # Example filtering function
    def mask_fn(row):
        return (
            True
            # & row["agent_type"] == "rflo"
            & (row["obs mask"].lower() == "first half")
            # & (row["obs mask"].lower() == "none")
            & (row["seed"] in [1, 2, 3, 4, 5])
            # & (row["env name"] in ["inverted pendulum", "ant", "halfcheetah", "reacher"])
        )
    
    df, latex = make_table(
        csv_path="data/eval/wandb_runs.csv",
        fields=FIELDS,
        val_field=VAL_FIELD,
        by=BY,
        index=INDEX,
        measure=MEASURE,
        sweeps=None,  # or SWEEPS
        mask_fn=mask_fn,
        sort_by_value=True,
        print_table=True,
        decimal_places=2,
        output_format="md",
    )
