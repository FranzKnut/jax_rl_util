"""Edit logged wandb runs in bulk.

Example
-------
>>> def add_commit(run, api):
...     run.config["commit"] = run.commit
>>> fix_runs("RTRRL_2", add_commit, sweep="xprah6y7", version=2)
"""

import argparse
from collections.abc import Callable

from jax_rl_util.eval.data import ENTITY


def fix_runs(
    project: str,
    fn: Callable,
    sweep: str | None = None,
    filters: dict | None = None,
    entity: str = ENTITY,
    version: int = 1,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Apply ``fn(run, api)`` to every run and mark it with ``config["version"]``.

    Runs that already have a ``version`` in their config are skipped unless
    ``force`` is set, so a fix can be re-run safely after a failure.

    Parameters
    ----------
    project : str
        wandb project, ``"name"`` or ``"entity/name"``.
    fn : Callable
        ``fn(run, api)`` edits ``run.config`` / ``run.summary`` in place.
    sweep : str, optional
        Only runs of this sweep.
    filters : dict, optional
        Extra wandb API filters.
    entity : str, optional
        Entity if ``project`` has none, by default "franzknut".
    version : int, optional
        Value written to ``config["version"]``, by default 1.
    force : bool, optional
        Also fix runs that already have a version, by default False.
    dry_run : bool, optional
        Call ``fn`` but do not save, by default False.

    Returns
    -------
    tuple[int, int, int]
        Number of fixed, skipped and failed runs.
    """
    import wandb
    from tqdm import tqdm

    api = wandb.Api()
    filters = dict(filters or {})
    if sweep is not None:
        filters["sweep"] = sweep
    path = project if "/" in project else f"{entity}/{project}"
    print("Fix version", version, "of", path, "with filters", filters)
    fixed = skipped = failed = 0
    pbar = tqdm(api.runs(path, filters=filters))
    for run in pbar:
        if run.config.get("version") is not None and not force:
            skipped += 1
            continue
        try:
            fn(run, api)
            run.config["version"] = version
            if not dry_run:
                run.update()
            fixed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            pbar.write(f"Failed to fix {run.name}: {e}")
    print(f"Fixed {fixed}, skipped {skipped}, failed {failed} runs.")
    return fixed, skipped, failed


def fix_runs_cli(fn: Callable, project: str, argv: list[str] | None = None):
    """Small CLI around ``fix_runs`` for use as a ``main(commands=...)`` entry.

    Parameters
    ----------
    fn : Callable
        The fix, see ``fix_runs``.
    project : str
        Default project.
    argv : list[str], optional
        Arguments: ``--project``, ``--sweep``, ``--version``, ``--force``, ``--dry-run``.

    Returns
    -------
    tuple[int, int, int]
        See ``fix_runs``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=project)
    parser.add_argument("--sweep", default=None)
    parser.add_argument("--version", default=1, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args(argv)
    return fix_runs(a.project, fn, a.sweep, version=a.version, force=a.force, dry_run=a.dry_run)
