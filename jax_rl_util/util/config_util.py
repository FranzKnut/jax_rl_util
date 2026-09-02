import json
import jax
import simple_parsing
from typing import Any


# json dict type taken from: https://github.com/decoderesearch/SAELens/blob/0a7aa47e7ea7b8d490c0af4d0ba9d2761cb339e2/sae_lens/config.py
# TODO: add tests (also copy from SAELens)
# calling this "json_dict" so error messages will reference "json_dict" being invalid
def json_dict(s: str) -> Any:
    try:
        res = json.loads(s)
    except json.JSONDecodeError as e:
        print()
        print(f"ERROR while parsing JSON string: {e}")
        print(f"Input string: {s}")
        print()
        raise e
    if res is not None and not isinstance(res, dict):
        raise ValueError(f"Expected a dictionary, got {type(res)}")
    return res


def dict_field(default: dict[str, Any] | None, **kwargs: Any) -> Any:  # type: ignore
    """
    Helper to wrap simple_parsing.helpers.dict_field so we can load JSON fields from the command line.
    """
    if default is None:
        return simple_parsing.helpers.field(default=None, type=json_dict, **kwargs)
    return simple_parsing.helpers.dict_field(default, type=json_dict, **kwargs)


def check_pytree_structure(tree1, tree2):
    """Checks if two parameter dictionaries have the same tree structure."""
    structure1 = jax.tree_util.tree_structure(tree1)
    structure2 = jax.tree_util.tree_structure(tree2)
    return structure1 == structure2
