"""Loop over all runs and do something."""
import argparse
import numpy as np
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
parser.add_argument("--sweep", default=None, type=str)
parser.add_argument("--version", default=1)
args = parser.parse_args()

filters = {}
if args.sweep is not None:
    filters['sweep'] = args.sweep

api = wandb.Api()
failed_count = 0
print("Fix version", args.version, "with filters", filters)
for r in tqdm(api.runs("datenvorsprung/TubeDAgger", filters=filters)):
    if r.config.get('version', None) is not None and not args.force:
        # skip if version is already set
        continue

    try:
        # DO SOMETHING TO RUNS HERE
        policy_acting = np.array([row["policy_acting"] for row in r.scan_history(keys=["policy_acting"])])
        policy_acting = policy_acting.reshape((-1, 1000))
        r.summary["computed_policy_acting"] = np.sum(policy_acting, axis=1)  # sum over steps
        r.summary["computed_context_switches"] = np.diff(policy_acting, axis=-1).sum(axis=-1)
    except:  # noqa
        failed_count += 1
        print(f"Failed to fix {r.name}, {failed_count} failed runs so far.")
    # DONE
    r.config['version'] = args.version
    r.update()

if failed_count > 0:
    print(f"Failed to fix {failed_count} runs.")
