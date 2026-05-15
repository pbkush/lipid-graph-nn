#!/bin/bash
# sbatch_benchmark_hpc.sh — SLURM worker for one benchmark sweep point.
#
# Runs SIMS_PER_NODE parallel gmx mdrun instances on a single node using a
# pre-built prun.tpr (produced by Phase 1 of benchmark_hpc.sh).  Slots cycle
# through the reference TPR list; GPU pinning mirrors sbatch_simulations.sh.
# All outputs land in BENCH_POINT_DIR so analyze_benchmark.py can find them.
#
# Env vars set by benchmark_hpc.sh at submission time:
#   BENCH_POINT_DIR   — directory to write slot logs and rocm-smi.tsv
#   REFERENCE_TPRS    — colon-separated tpr paths (cycled across slots)
#   SIMS_PER_NODE     — number of parallel slots
#   GPUS_PER_NODE     — 0 = CPU mode; >0 = HIP_VISIBLE_DEVICES pinning
#   CPUS_PER_SIM      — threads per slot (informational; ntomp derived from SLURM)
#   NSTEPS            — gmx mdrun -nsteps value

#SBATCH --job-name=lipid-bench
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --mail-type=FAIL
#SBATCH --account=cellmembrane
#SBATCH --output=logs/simulations/bench-%j.out
#SBATCH --error=logs/simulations/bench-%j.err

set -euo pipefail
mkdir -p logs/simulations

source "$HOME/miniforge3/etc/profile.d/conda.sh"
cd "$HOME/lipid-graph-nn"

CONDA_ENV=$(python scripts/python/print_config_var.py hpc.conda_env)
conda activate "$CONDA_ENV"

MODULEFILES_PATH=$(python scripts/python/print_config_var.py hpc.modulefiles_path)
MODULE_GROMACS=$(python scripts/python/print_config_var.py hpc.module_gromacs)
module purge
module use "$MODULEFILES_PATH"
module load "$MODULE_GROMACS"

: "${BENCH_POINT_DIR:?BENCH_POINT_DIR must be set by benchmark_hpc.sh}"
: "${REFERENCE_TPRS:?REFERENCE_TPRS must be set by benchmark_hpc.sh}"
: "${SIMS_PER_NODE:?SIMS_PER_NODE must be set by benchmark_hpc.sh}"

N_SIMS="$SIMS_PER_NODE"
mkdir -p "$BENCH_POINT_DIR"

# Parse colon-separated TPR list into an array
IFS=: read -ra TPRS_ARR <<< "$REFERENCE_TPRS"
N_TPRS=${#TPRS_ARR[@]}

NTOMP_VALUE=$(( SLURM_CPUS_PER_TASK / N_SIMS ))

echo "Benchmark point on $(hostname)"
echo "  point dir  : $BENCH_POINT_DIR"
echo "  sims       : $N_SIMS"
echo "  gpus       : ${GPUS_PER_NODE:-0}"
echo "  ntomp/sim  : $NTOMP_VALUE"
echo "  nsteps     : ${NSTEPS:-100000}"
echo ""

# Background rocm-smi sampler (GPU mode only; skips gracefully if not on PATH)
ROCM_SMI_PID=-1
if command -v rocm-smi &>/dev/null && [[ "${GPUS_PER_NODE:-0}" -gt 0 ]]; then
    (
        printf 'timestamp\tgpu_id\tutilization_pct\tpower_W\tvram_used_MB\n'
        while true; do
            rocm-smi --showuse --showpower --showmeminfo vram --csv 2>/dev/null \
                | awk -v ts="$(date +%s)" \
                    'NR>1 && NF>1 { printf "%s\t%s\n", ts, $0 }' \
                || true
            sleep 5
        done
    ) > "$BENCH_POINT_DIR/rocm-smi.tsv" &
    ROCM_SMI_PID=$!
fi

PIDS=()

for (( i=0; i<N_SIMS; i++ )); do
    TPR="${TPRS_ARR[$(( i % N_TPRS ))]}"
    SLOT_DIR="$BENCH_POINT_DIR/slot_${i}"
    mkdir -p "$SLOT_DIR"

    # -nstxout/-nstvout/etc. are MDP options, not mdrun CLI flags; the TPR
    # already baked in the output frequency from prun.mdp.  Trajectory I/O
    # over a 100k-step benchmark is a negligible perf contributor.
    #
    # gmx v2025.4 requires -ntmpi when -ntomp is set on GPU runs.  Each slot
    # is one process with one GPU → -ntmpi 1.
    MDRUN_ARGS=(
        gmx mdrun
        -s    "$TPR"
        -deffnm bench
        -nsteps "${NSTEPS:-100000}"
        -ntomp  "$NTOMP_VALUE"
        -resethway
    )
    if [[ "${GPUS_PER_NODE:-0}" -eq 0 ]]; then
        MDRUN_ARGS+=(-nb cpu)
    else
        MDRUN_ARGS+=(-ntmpi 1)
    fi

    echo "  [slot $i]  $(basename "$(dirname "$TPR")")  → $SLOT_DIR/bench.log"

    (
        if [[ "${GPUS_PER_NODE:-0}" -gt 0 ]]; then
            export HIP_VISIBLE_DEVICES="$(( i % GPUS_PER_NODE ))"
            export CUDA_VISIBLE_DEVICES="$(( i % GPUS_PER_NODE ))"
        fi
        export OMP_NUM_THREADS="$NTOMP_VALUE"
        cd "$SLOT_DIR"
        "${MDRUN_ARGS[@]}"
    ) &
    PIDS+=($!)
done

echo ""

EXIT_CODE=0
for (( i=0; i<${#PIDS[@]}; i++ )); do
    if wait "${PIDS[$i]}"; then
        echo "  [slot $i] OK"
    else
        rc=$?
        echo "  [slot $i] FAILED  exit=$rc"
        EXIT_CODE=$rc
    fi
done

# Stop rocm-smi sampler
[[ "$ROCM_SMI_PID" -ne -1 ]] && kill "$ROCM_SMI_PID" 2>/dev/null || true

echo ""
echo "Benchmark point done on $(hostname)  (exit_code=$EXIT_CODE)"
exit "$EXIT_CODE"
