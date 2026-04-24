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

GROUP=cellmembrane
USER=pberger

WORK="/work/${GROUP}/${USER}/lipid_data/colab_lipid_gnn_subset/processed"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lipid_gnn
module load rocm/7.2.0

# Stage chunks to fast node-local storage once per job.
# Copies the full processed/ tree (train/, val/, test/ subdirs).
STAGE="/local/${SLURM_JOB_ID}"
echo "Staging chunks to $STAGE ..."
mkdir -p "$STAGE"
rsync -a "$WORK/" "$STAGE/"
export CHUNKS_DIR="$STAGE"

# W&B: set WANDB_MODE=offline in your environment if compute nodes are air-gapped;
# then after the job: wandb sync "$WORK"/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="stage_2_weight_decay-sweep"
export WANDB_DIR="/home/cellmembrane/pberger/lipid-graph-nn/results/wandb"
mkdir -p "$WANDB_DIR"

cd "$HOME/lipid-graph-nn"
python scripts/training/run_sweep.py
