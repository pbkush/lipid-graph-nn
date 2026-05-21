#!/bin/bash
# sbatch_sweep_serial.sh — entry point for one SLURM job; runs a single
# training process (one GPU, multiple seeds sequentially inside run_sweep.py).
#
# Counterpart to sbatch_sweep.sh, which packs N parallel runs per node with
# --gres=gpu:N. This serial variant:
#   * does NOT use --gres (set via submit_sweep_serial.sh on the sbatch
#     command line if/when the target partition needs it)
#   * runs run_sweep.py once per job
#   * keeps config-default DataLoader settings (num_workers=6, pin_memory=1) —
#     no IPC contention because there is no per-node fan-out
#   * lets run_sweep.py iterate the SWEEP grid serially when multiple seeds
#     are bundled into SWEEP_SEEDS
#
# --partition, --time, --cpus-per-task and --mem are set on the sbatch
# command line by submit_sweep_serial.sh; no static #SBATCH directives for
# them here.

#SBATCH --job-name=lipid-sweep
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --account=cellmembrane
#SBATCH --output=logs/sweeps/sweep-%j.out
#SBATCH --error=logs/sweeps/sweep-%j.err

set -euo pipefail
mkdir -p logs/sweeps

source "$HOME/miniforge3/etc/profile.d/conda.sh"

cd "$HOME/lipid-graph-nn"

CONDA_ENV=$(python scripts/python/print_config_var.py hpc.conda_env)
conda activate "$CONDA_ENV"
module load "$(python scripts/python/print_config_var.py hpc.module_rocm)"

USER=pberger
GROUP_HPC="$(python scripts/python/print_config_var.py hpc.group)"
WORK_SUBPATH="$(python scripts/python/print_config_var.py hpc.work_subpath)"
# On HPC the preprocessed graphs live flat under preprocessed_graphs/ (the
# train/val/test split dirs are direct children) — no swap-able 'active'
# subdir like the local layout has.
WORK="/work/${GROUP_HPC}/${USER}/${WORK_SUBPATH}/preprocessed_graphs"

# W&B: set WANDB_MODE=offline in your environment if compute nodes are air-gapped;
# then after the job: wandb sync "$WORK"/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_GROUP="${WANDB_GROUP:-stage_0_baseline}"
WANDB_DIR="$(python scripts/python/print_config_var.py paths.wandb_dir)"
export WANDB_DIR
mkdir -p "$WANDB_DIR"

# Single-process job: GPFS direct read with config-default num_workers (6) and
# pin_memory. Threaded prefetch inside MartiniDiskDataset covers GPU/IO overlap.
# CHUNKS_DIR can be overridden by submit_sweep_serial.sh via --chunks-dir to
# pick a specific labels-set subdir (e.g. .../preprocessed_graphs/<set>/).
export CHUNKS_DIR="${CHUNKS_DIR:-$WORK}"
echo "Launching 1 training process on $(hostname)"
echo "  Chunks : $CHUNKS_DIR (GPFS direct)"
echo "  Workers: (config default — typically 6)"

python scripts/training/run_sweep.py
