"""Get results from wandb and make latex table."""

import os

import pandas as pd

import wandb

api = wandb.Api()

# PROJECTS = ["brax_imitation"]
PROJECTS = ["RTRRL", "neurips24", "gymnax_new", "brax_new", "AC_brax", "AC_gymnax", "PPO_Gymnax"]

SWEEPS = None  # None for all sweeps


def get_runs_for_config(project, filters={}):
    """Get all runs for a config."""
    # Project is specified by <entity/project-name>
    runs = api.runs(project, filters=filters)

    summaries = []
    for run in runs:
        
        env_name = run.config["env_name"] if "env_name" in run.config else run.config["env_params"]["env_name"]

        summaries.append(
            {
                "name": run.name,
                "env_name": env_name,
                "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
                "Sweep": run.sweep.id if run.sweep is not None else "none",
                "created_at": pd.to_datetime(run.created_at),
                **run.summary._json_dict,  # .summary contains the output keys/values for metrics like accuracy. We call ._json_dict to omit large files
            }
        )

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Get all runs for a config
    all_dfs = []
    for p in PROJECTS:
        _p = api.project(p, entity="franzknut")
        print("Getting sweeps for project", p)

        if SWEEPS is None:
            sweep_runs = get_runs_for_config(p)
            all_dfs.append(sweep_runs)
        else:
            all_sweeps = {s.id: s.name for s in _p.sweeps()}  # if s.name[:2] in ["32"]}
            for s in all_sweeps.keys():
                print(all_sweeps[s])
                filters = {
                    "Sweep": s,
                }
                sweep_runs = get_runs_for_config(p, filters)
                sweep_runs["Sweep_name"] = all_sweeps[s]
                all_dfs.append(sweep_runs)

    df_out = pd.concat(all_dfs)
    print("downloaded:")
    print(df_out)
    # Save to csv
    print(df_out)
    os.makedirs("eval/data", exist_ok=True)
    df_out.to_csv("eval/data/wandb_runs.csv")
