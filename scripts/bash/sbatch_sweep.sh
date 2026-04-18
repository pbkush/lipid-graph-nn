#!/bin/bash
# Run the training sweep on a Goethe-HLR MI210.
# Dev/tuning (1 GPU, 8 h): sbatch scripts/bash/sbatch_sweep.sh
# Full node (8 GPUs, longer): edit partition+gres below or pass via sbatch flags.
#
# Requires: env GROUP=<goethe-group>, chunks prebuilt via sbatch_preprocess.sh.

#SBATCH --job-name=lipid-sweep
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err

set -euo pipefail
mkdir -p logs

: "${GROUP:?set GROUP to your Goethe-HLR group (e.g. export GROUP=fias)}"
WORK="/work/${GROUP}/${USER}/lipid-data"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lipid_gnn
module load rocm/6.2.4

# Stage chunks to fast node-local storage once per job.
STAGE="/local/${SLURM_JOB_ID}/chunks"
echo "Staging chunks to $STAGE ..."
mkdir -p "$STAGE"
cp "$WORK"/chunks/chunk_*.pt "$STAGE/"
export CHUNKS_DIR="$STAGE"

# W&B: set WANDB_MODE=offline in your environment if compute nodes are air-gapped;
# then after the job: wandb sync "$WORK"/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$WORK/wandb"
mkdir -p "$WANDB_DIR"

cd "$HOME/lipid-graph-nn"
python scripts/training/run_sweep.py
