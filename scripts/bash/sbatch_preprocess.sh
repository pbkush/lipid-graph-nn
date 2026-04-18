#!/bin/bash
# Preprocess Martini trajectories on Goethe-HLR into .pt chunks on /work.
# Submit from the repo root: sbatch scripts/bash/sbatch_preprocess.sh
#
# Before first use: set GROUP to your Goethe-HLR group directory and confirm
# the raw data has been rsynced to $WORK/lipid-data/data/membrane_only/ and
# property .h5 files to $WORK/lipid-data/results/properties/.

#SBATCH --job-name=lipid-preprocess
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err

set -euo pipefail
mkdir -p logs

: "${GROUP:?set GROUP to your Goethe-HLR group (e.g. export GROUP=fias)}"
WORK="/work/${GROUP}/${USER}/lipid-data"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lipid_gnn

cd "$HOME/lipid-graph-nn"

python scripts/training/prepare_colab_subset.py \
    --no-zip \
    --sims-dir  "$WORK/data/membrane_only" \
    --props-dir "$WORK/results/properties" \
    --out-dir   "$WORK/chunks" \
    --properties lipid_packing thickness \
    --num-frames 50 \
    --chunk-size 50 \
    --spatial-cutoff 9.0
