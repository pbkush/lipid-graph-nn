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
CHUNKS_REL="$(python scripts/python/print_config_var.py paths.chunks_dir)"
# paths.chunks_dir resolves absolute against REPO_ROOT; strip to get the tail.
CHUNKS_BASENAME="$(basename "$(dirname "$CHUNKS_REL")")/$(basename "$CHUNKS_REL")"
WORK="/work/${GROUP_HPC}/${USER}/${WORK_SUBPATH}/${CHUNKS_BASENAME}"

# Stage chunks to fast node-local storage once per job (shared by all
# parallel processes on this node).
STAGE="/local/${SLURM_JOB_ID}"
echo "Staging chunks from $WORK to $STAGE ..."
mkdir -p "$STAGE"
rsync -a "$WORK/" "$STAGE/"
export CHUNKS_DIR="$STAGE"

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

# Scale DataLoader workers down so the total worker-process count on the node
# stays bounded. 4 runs × 6 workers = 24 concurrent workers share the same
# /dev/shm and multiprocessing IPC sockets, which overloads the resource
# sharer and causes "Pin memory thread exited unexpectedly" / FileNotFoundError.
# Integer division floors at 0, which is valid (synchronous loading in main
# process — safe and often fast enough from node-local staging).
DEFAULT_NUM_WORKERS=$(python scripts/python/print_config_var.py training.num_workers)
PER_GPU_WORKERS=$(( DEFAULT_NUM_WORKERS / N_RUNS ))
export FREEZE_NUM_WORKERS="$PER_GPU_WORKERS"
echo "  DataLoader workers: ${DEFAULT_NUM_WORKERS} config → ${PER_GPU_WORKERS}/slot (${N_RUNS} slots)"

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
