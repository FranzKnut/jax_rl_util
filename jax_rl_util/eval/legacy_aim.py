"""Helpers for old aim repos. Logging moved to wandb; kept for old results.

Requires ``aim`` and, for ``aim_lineplot``, ``plotly``.
"""

from collections.abc import Callable

import numpy as np


def aim_lineplot(repos: dict, envs: list[str], colors: dict | None = None, obs_mask: str = "even"):
    """Plot the best eval curve per env for each aim repo (one subplot per env).

    Parameters
    ----------
    repos : dict
        Label -> aim repo path, e.g. ``{"RTRRL": "aim_repos/brax_new/"}``.
    envs : list[str]
        Env names like ``"brax-ant"``.
    colors : dict, optional
        Label -> color.
    obs_mask : str, optional
        Only runs with this obs mask, by default "even".

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import aim
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=len(envs), subplot_titles=envs)
    for label, path in repos.items():
        repo = aim.Repo(path)
        for col, env in enumerate(envs):
            short = env.split("-")[1] if "-" in env else env
            query = (
                f"(run.hparams.env_params.env_name == '{env}' or run.hparams.env_name == '{short}')"
                f" and (run.hparams.env_params.obs_mask == '{obs_mask}' or run.hparams.obs_mask == '{obs_mask}')"
                " and (metric.name == 'eval/rewards' or metric.name == 'eval/episode_reward')"
            )
            best, x = [-np.inf], [0]
            for run in repo.query_metrics(query).iter_runs():
                for metric in run:
                    values = metric.values.values_numpy()
                    if max(values) > max(best):
                        best, x = values, metric.epochs.values_list()
            fig.add_trace(
                go.Scatter(y=best, x=sorted(x), name=label, marker=dict(color=(colors or {}).get(label)),
                           showlegend=col == 0),
                row=1, col=col + 1,
            )
            fig.update_xaxes(title_text="Epochs", row=1, col=col + 1)
            fig.update_yaxes(title_text="Episode Return", row=1, col=col + 1)
    return fig


def delete_aim_runs(repo: str = ".", query: str = "run.created_at.day < 10", remove: bool = False) -> list[str]:
    """Find runs matching ``query`` and delete them if ``remove`` is set.

    Parameters
    ----------
    repo : str, optional
        aim repo path, by default ".".
    query : str, optional
        aim run query.
    remove : bool, optional
        Actually delete, by default False (only list).

    Returns
    -------
    list[str]
        Hashes of the matching runs.
    """
    import aim

    r = aim.Repo(repo)
    hashes = [x.run.hash for x in r.query_runs(query).iter_runs()]
    print(f"Found {len(hashes)} runs")
    if remove:
        r.delete_runs(hashes)
        print("Removed.")
    return hashes


def fix_aim_runs(repo: str, fn: Callable[[dict], dict], query: str = "", version: int = 1, force: bool = False):
    """Apply ``fn(hparams) -> hparams`` to all matching runs and set ``fix_version``.

    Parameters
    ----------
    repo : str
        aim repo path.
    fn : Callable
        Edits the hparams dict and returns it.
    query : str, optional
        aim run query, by default all runs.
    version : int, optional
        Value for ``run["fix_version"]``, by default 1.
    force : bool, optional
        Also edit runs that already have a fix_version.

    Returns
    -------
    int
        Number of failed runs.
    """
    import aim
    from tqdm import tqdm

    r = aim.Repo(repo)
    hashes = [x.run.hash for x in r.query_runs(query).iter_runs() if force or x.run.get("fix_version") is None]
    failed = 0
    for h in tqdm(hashes):
        try:
            run = aim.Run(h, repo=repo)
            run["fix_version"] = version
            run["hparams"] = fn(run["hparams"])
            run.close()
        except Exception as e:  # noqa: BLE001
            failed += 1
            tqdm.write(f"Failed {h}: {e}")
    print(f"Done, {failed} failed.")
    return failed
