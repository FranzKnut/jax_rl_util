#!/bin/bash

#SBATCH -J "jax_rl_util"
## SBATCH --gpus=2
#SBATCH --output=./slurm_out/%x-%A_%a.out

# MAKE SURE OUTPUT DIR EXISTS!!

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 

# Set NVCC tmp directory
export TMPDIR=/scratch/tmp  # e.g., $HOME/tmp or /scratch
mkdir -p $TMPDIR

source ~/.bashrc
micromamba activate jax
source run.sh

echo $@
$@
