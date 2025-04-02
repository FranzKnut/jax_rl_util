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

        return pd.Series({n: deep_get(cfg, n) for n in names})

    df["config"] = df.config.apply(eval)
    return df.assign(**df.config.apply(_pull_fields))
