#!/usr/bin/env bash
# Launch wandb sweep agents across the SLURM servers in slurm_servers.txt.
#
#   ./launch_batch_slurm.sh [--no-prep] <entity/project/sweep_id> [jobs_per_server]
#
# --no-prep skips the git pull / poetry update step on the servers.
#
# `make_sweep_policy.py` prints the sweep path when it creates a sweep. `model`
# only names the SLURM jobs.

repo_dir="jax_rl_util"
# micromamba, not mamba: the alias isn't expanded in non-interactive bash
install="git clone git@github.com:FranzKnut/jax_rl_util.git && cd jax_rl_util && source ~/.bashrc && micromamba activate jax && poetry update"
prep="git pull && source ~/.bashrc && micromamba activate jax && poetry update && mkdir -p slurm_out"
sbatch_script="job_datalab.sh"

skip_prep=0
args=()
for arg in "$@"; do
    case "$arg" in
        --no-prep) skip_prep=1 ;;
        *) args+=("$arg") ;;
    esac
done
set -- "${args[@]}"

cmd=${1:?usage: ./launch_batch_slurm.sh [--no-prep] <command> [jobs_per_server]}
num_tasks=${2:-1}

launch="$sbatch_script python -m $cmd"
run_id=$(date +%Y%m%d_%H%M%S)

# Snapshot server list once to avoid races if the file changes mid-run.
servers=("datalab")
# while IFS= read -r server || [ -n "$server" ]; do
#     [ -z "$server" ] && continue
#     if [[ "$server" == \#* ]]; then
#         continue
#     fi
#     servers+=("$server")
# done < slurm_servers.txt

if (( ${#servers[@]} == 0 )); then
    echo "No valid servers found in slurm_servers.txt"
    exit 1
fi

for server in "${servers[@]}"; do
    servers_list+="$server, "
done

# Remove trailing comma from servers_list
servers_list=${servers_list%, }

echo "Command:  $cmd"
(( skip_prep )) && echo "Skipping prep"
read -r -p "Launch $num_tasks agents on: $servers_list? (y/n) " answer
if [[ ! $answer == "y" ]]; then
    exit 0
fi

for server in "${servers[@]}"; do
    echo

    ssh "$server" "env REPO_DIR='$repo_dir' RUN_ID='$run_id' NUM_TASKS='$num_tasks' INSTALL='$install' PREP='$prep' SKIP_PREP='$skip_prep' LAUNCH='$launch' SWEEP_ID='$cmd' bash -s" <<'EOF'
# set -euo pipefail
export PATH="~/.local/bin:$PATH"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo
    echo $INSTALL
    eval "$INSTALL" || { echo "Install failed, aborting."; exit 1; }
fi

cd "$REPO_DIR" || exit 1
echo
echo "Current directory: $(pwd)"
if [ "$SKIP_PREP" = 1 ]; then
    echo "Skipping prep"
else
    echo $PREP
    eval "$PREP" || { echo "Prep failed, aborting."; exit 1; }
fi

for i in $(seq 1 "$NUM_TASKS"); do
    echo
    echo "Launching agent $i"
    session_name="${REPO_DIR}_${SWEEP_ID##*/}_${RUN_ID}_${i}"
    mkdir -p slurm_out
    echo sbatch --job-name="$session_name" $LAUNCH
    sbatch --job-name="$session_name" $LAUNCH
done

echo
squeue -u $USER
EOF
    (( $? == 0 )) || { echo "Aborting: setup failed on $server"; exit 1; }
done
