"""Get results from wandb and make latex table."""

import os

import pandas as pd
import wandb
from tqdm import tqdm

api = wandb.Api()

PROJECTS = ["TubeDAgger"]
# PROJECTS = ["RTRRL", "neurips24", "gymnax_new", "brax_new", "AC_brax", "AC_gymnax", "PPO_Gymnax"]
# PROJECTS = ["RTRRL"]

SWEEPS = None  # None for all sweeps


def get_runs_for_config(project, filters={}):
    """Get all runs for a config."""
    # Project is specified by <entity/project-name>
    runs = api.runs(project, filters=filters, per_page=100, lazy=False)
    print("Found", len(runs), "runs")
    summaries = []
    for run in tqdm(runs):
        summaries.append(
            {
                "name": run.name,
                "config": {
                    k: v for k, v in run.config.items() if not k.startswith("_")
                },
                "Sweep": run.sweep.id if run.sweep is not None else "none",
                "created_at": pd.to_datetime(run.created_at),
                **run.summary._json_dict,  # .summary contains the output keys/values for metrics like accuracy. We call ._json_dict to omit large files
            }
        )

    return pd.DataFrame(summaries)


def wandb_to_csv(projects: list[str], sweeps: list[str] = None, out_dir: str = "data/eval", force: bool = False):
    """Download wandb runs for given projects and save them as csv.

    Parameters
    ----------
    projects : list[str]
        List of projects to download runs for
    sweeps : list[str], optional
        Only download runs for these sweeps, by default all sweeps are downloaded
    out_dir : str, optional
        Path to save the output CSV files, by default "data/eval"
    force : bool, optional
        Whether to overwrite existing files, by default False
    """
    os.makedirs(out_dir, exist_ok=True)
    for p in projects:
        _p = api.project(p, entity="franzknut")
        print("Getting sweeps for project", p)

        if sweeps is None:
            sweep_runs = get_runs_for_config(p)
            for sweep_id, group_df in sweep_runs.groupby("Sweep"):
                out_path = os.path.join(out_dir, f"{sweep_id}.csv")
                if os.path.exists(out_path) and not force:
                    print(f"Skipping {sweep_id}, file already exists.")
                    continue
                group_df.to_csv(out_path)
                print("Saved to", out_path)
        else:
            all_sweeps = {s.id: s.name for s in _p.sweeps() if s.id in sweeps}
            for s in all_sweeps.keys():
                out_path = os.path.join(out_dir, f"{s}.csv")
                if os.path.exists(out_path) and not force:
                    print(f"Skipping {s}, file already exists.")
                    continue
                
                print(all_sweeps[s])
                filters = {
                    "Sweep": s,
                }
                sweep_runs = get_runs_for_config(p, filters)
                sweep_runs["Sweep_name"] = all_sweeps[s]
                sweep_runs.to_csv(out_path)
                print("Saved to", out_path)


if __name__ == "__main__":
    wandb_to_csv()
