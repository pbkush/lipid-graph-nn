#!/bin/bash
# sbatch_sweep.sh — entry point for one SLURM job; fans out N parallel
# training processes (one per GPU) on the allocated node.
#
# --partition, --time, --gres, --cpus-per-task and --mem are set on the sbatch
# command line by submit_sweep.sh, so no static #SBATCH directives for them
# here. Per-run hyperparameters arrive as RUN_<i>_* env vars; N_RUNS_PER_NODE
# tells us how many slots to launch.

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

# ── Launch N parallel training processes, one per GPU ────────────────────────
N_RUNS="${N_RUNS_PER_NODE:-1}"
echo "Launching $N_RUNS parallel training run(s) on $(hostname)"

# All jobs: read chunks directly from GPFS — no staging to /local.
#
# Root cause of all prior DataLoader failures: /local ($SLURM_JOB_ID) is a
# tmpfs. Its pages get evicted under memory pressure (from IPC shared-memory
# of DataLoader workers, or from competing jobs on shared gpu_test nodes),
# turning file reads into FileNotFoundError. GPFS files can never be evicted
# from the filesystem — slow page-cache misses at worst, never a missing file.
#
# I/O overlap with GPU compute is recovered by a background thread inside
# MartiniDiskDataset.__iter__ (num_workers=0 path): while the GPU processes
# chunk N, the thread reads chunk N+1 from GPFS. Threads share the main
# process memory — no child processes, no IPC sockets, nothing to OOM-kill.
export CHUNKS_DIR="$WORK"
export FREEZE_NUM_WORKERS=0
export FREEZE_PIN_MEMORY=0
echo "  Chunks : $WORK (GPFS — no staging, threaded prefetch)"
echo "  Workers: 0/slot (IPC-free)"

PIDS=()
for ((i=0; i<N_RUNS; i++)); do
    LOGOUT="logs/sweeps/sweep-${SLURM_JOB_ID}-gpu${i}.out"
    LOGERR="logs/sweeps/sweep-${SLURM_JOB_ID}-gpu${i}.err"

    # Indirect lookup of per-slot env vars set by submit_sweep.sh.
    H_VAR="RUN_${i}_HIDDEN_DIM";  H="${!H_VAR:-}"
    L_VAR="RUN_${i}_NUM_LAYERS";  L="${!L_VAR:-}"
    LR_VAR="RUN_${i}_LR";         LR="${!LR_VAR:-}"
    WD_VAR="RUN_${i}_WD";         WD="${!WD_VAR:-}"
    SD_VAR="RUN_${i}_SEED";       SD="${!SD_VAR:-}"

    echo "  [GPU $i] h=$H l=$L lr=$LR wd=$WD seed=${SD:-"(default)"}  → $LOGOUT"

    (
        # Pin this process to a single GPU. ROCm honours HIP_VISIBLE_DEVICES;
        # CUDA_VISIBLE_DEVICES is set too for portability.
        export CUDA_VISIBLE_DEVICES="$i"
        export HIP_VISIBLE_DEVICES="$i"

        [[ -n "$H"  ]] && export FREEZE_HIDDEN_DIM="$H"
        [[ -n "$L"  ]] && export FREEZE_NUM_LAYERS="$L"
        [[ -n "$LR" ]] && export FREEZE_LR="$LR"
        [[ -n "$WD" ]] && export FREEZE_WD="$WD"
        [[ -n "$SD" ]] && export SWEEP_SEEDS="$SD"

        python scripts/training/run_sweep.py >"$LOGOUT" 2>"$LOGERR"
    ) &
    PIDS+=($!)
done

# Wait for every parallel process; surface the worst exit code.
EXIT_CODE=0
for ((i=0; i<${#PIDS[@]}; i++)); do
    if ! wait "${PIDS[$i]}"; then
        rc=$?
        echo "  [GPU $i] FAILED (exit $rc)"
        EXIT_CODE=$rc
    else
        echo "  [GPU $i] OK"
    fi
done

echo "All runs done (exit_code=$EXIT_CODE)"
exit "$EXIT_CODE"
