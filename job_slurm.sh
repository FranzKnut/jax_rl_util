#!/bin/bash

#SBATCH -J "RTRRL"
#SBATCH --partition=zen3_0512
#SBATCH --ntasks=10
#SBATCH --qos=zen3_0512
#SBATCH --output=./slurm_out/%x-%A_%a.out

# MAKE SURE OUTPUT DIR EXISTS!!

echo 
echo 'Hello from node: '$HOSTNAME
echo 'Number of nodes: '$SLURM_JOB_NUM_NODES
echo 'Tasks per node:  '$SLURM_TASKS_PER_NODE
echo 'Partition used:  '$SLURM_JOB_PARTITION
echo 'Using the nodes: '$SLURM_JOB_NODELIST
echo 

source run.sh
$@
