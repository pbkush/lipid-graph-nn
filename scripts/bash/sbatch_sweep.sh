#!/bin/bash

#SBATCH --job-name=stage_2_lipid-sweep
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --account=cellmembrane
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/sweeps/sweep-%j.out
#SBATCH --error=logs/sweeps/sweep-%j.err

set -euo pipefail
mkdir -p logs/sweeps

source "$HOME/miniforge3/etc/profile.d/conda.sh"

cd "$HOME/lipid-graph-nn"

CONDA_ENV=$(python scripts/python/print_config_var.py hpc.conda_env)
conda activate "$CONDA_ENV"
module load "$(python scripts/python/print_config_var.py hpc.module_rocm)"

GROUP="$(python scripts/python/print_config_var.py hpc.group)"
WORK_SUBPATH="$(python scripts/python/print_config_var.py hpc.work_subpath)"
CHUNKS_REL="$(python scripts/python/print_config_var.py paths.chunks_dir)"
# paths.chunks_dir resolves absolute against REPO_ROOT; strip to get the tail.
CHUNKS_BASENAME="$(basename "$(dirname "$CHUNKS_REL")")/$(basename "$CHUNKS_REL")"
WORK="/work/${GROUP}/${USER}/${WORK_SUBPATH}/${CHUNKS_BASENAME}"

# Stage chunks to fast node-local storage once per job.
# Copies the full processed/ tree (train/, val/, test/ subdirs).
STAGE="/local/${SLURM_JOB_ID}"
echo "Staging chunks from $WORK to $STAGE ..."
mkdir -p "$STAGE"
rsync -a "$WORK/" "$STAGE/"
export CHUNKS_DIR="$STAGE"

# W&B: set WANDB_MODE=offline in your environment if compute nodes are air-gapped;
# then after the job: wandb sync "$WORK"/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="stage_2_weight_decay-sweep"
WANDB_DIR="$(python scripts/python/print_config_var.py paths.wandb_dir)"
export WANDB_DIR
mkdir -p "$WANDB_DIR"

python scripts/training/run_sweep.py
