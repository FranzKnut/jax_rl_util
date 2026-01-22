"""Get results from wandb and make latex table."""

import os
import pandas as pd
from tqdm import tqdm
import wandb
from wandb.apis.public.runs import Run

api = wandb.Api(timeout=20)

all_sweeps = ["kn8bcn7x"]
# envs = all_env_names
# envs = ["StatelessCartPoleEasy"]
projects = ["RTRRL"]

# extra_filter = {"config.hidden_size": 64, "config.rnn_model": "ctrnn"}
extra_filter = {"config.hidden_size": 64, "config.rnn_model": "ctrnn"}
extra_filter = {}
dataset_name = "".join([str(s) for s in projects + all_sweeps + list(extra_filter.values())])


def get_runs_for_config(project, filters={}, max_step=1e7):
    """Get all runs for a config."""
    # Project is specified by <entity/project-name>
    runs = api.runs(project, filters=filters)

    best_rewards, datas, env_names, name_list, config_list, sweep_list = [], [], [], [], [], []
    for run in tqdm(runs):
        run: Run
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        env_name = run.config["env_params"]["env_name"]
        if "best_eval_reward" not in run.summary._json_dict:
            continue

        best_rewards.append(run.summary._json_dict["best_eval_reward"])
        history = run.scan_history(keys=["mean_reward", "_step"], max_step=max_step, page_size=max_step)
        data = pd.DataFrame(history)
        datas.append(data.unstack())

        env_names.append(env_name)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        sweep_list.append(run.sweep.id)

    df = pd.DataFrame(
        {
            "name": name_list,
            "env_name": env_names,
            "best_eval_reward": best_rewards,
            "config": config_list,
            "Sweep": sweep_list,
        }
    )
    return pd.concat(
        [
            df,
            pd.DataFrame(datas, index=df.index),
        ],
        axis=1,
    )


if __name__ == "__main__":
    # Get all runs for a config
    all_dfs = []
    for p in projects:
        for s in all_sweeps:
            filters = {"Sweep": s, **extra_filter}
            all_dfs.append(get_runs_for_config("franzknut/" + p, filters))
    df = pd.concat(all_dfs)
    print("downloaded:")
    # Save to csv
    print(df)
    os.makedirs("eval/data", exist_ok=True)
    df.to_csv(f"eval/data/wandb_runs_history{dataset_name}.csv")
