import pandas as pd


def gen_dict_extract(var, key):
    if hasattr(var, "items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(v, key):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(d, key):
                        yield result


def deep_get(dictionary, keys, default=None):
    generator = gen_dict_extract(dictionary, keys)
    try:
        return next(generator)
    except StopIteration:
        return default


def combine_mean_pm_std(df, metric="mean"):
    """Convert mean and std to latex string."""
    return df.apply(lambda x: f"{x[metric]:.2f}$\\pm${x['std']:.2f}", axis=1)


def combine_mean_pm_std_multi(x, metric="mean"):
    """Convert mean and std to latex string when working with multi-index."""
    return (
        x[metric].map(lambda a: f"{a:.2f}")
        + "$\pm$"
        + x["std"].map(lambda a: f"{a:.2f}")
    )


def pull_fields(df: pd.DataFrame, names: list[str] = []):
    """Get fields with given names from config column and extract to separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that hase a column called "config" that is a dictionary.
        The column may also contain a string that encodes a dictionary
    names : list[str], optional
        List of names for fields to extraxt to separate columns, by default []
    """

    def _pull_fields(cfg):
        """Pull relevant fields from the config field."""
        if isinstance(cfg, str):
            cfg = eval(cfg)

        fields = {n: deep_get(cfg, n) for n in names}
        for k, v in fields.items():
            try:
                hash(v)
            except:
                if isinstance(v, list):
                    fields[k] = tuple(v)
                else:
                    fields[k] = str(v)
        return pd.Series(fields)

    if df.config.dtype == str:
        df["config"] = df.config.apply(eval)
    return df.assign(**df.config.apply(_pull_fields))


def print_latex_table(df: pd.DataFrame, max_num_cols=3):
    print(df)
    print("")
    print("")
    for i in range(df.shape[1] // max_num_cols):
        print(
            df.iloc[:, i * max_num_cols : (i + 1) * max_num_cols].to_latex(
                escape=False,
                column_format="l" + "c" * (df.shape[-1]),
                multicolumn_format="c",
                float_format="{:.2f}".format,
                # header=headers,
            )
        )
