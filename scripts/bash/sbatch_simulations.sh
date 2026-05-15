#!/bin/bash
# sbatch_simulations.sh — SLURM worker for Martini 3 bilayer simulations.
#
# Runs inside a SLURM allocation submitted by submit_simulations.sh.
# Fans out N parallel run_martini_pipeline.py processes (one per GPU slot, or
# CPU-distributed when GPUS_PER_NODE=0), waits for all to complete, and exits
# with the worst per-slot exit code.
#
# All simulation parameters arrive as environment variables baked in by the
# orchestrator at submission time:
#   OUTPUT_ROOT       — top-level output directory (on /work)
#   N_SIMS_PER_NODE   — number of parallel simulations to run on this node
#   RUN_<i>_COMP      — canonical composition name for slot i
#   PROD_NS or NSTEPS — production length
#   MAXWARN           — gmx grompp -maxwarn value
#   SAVE_FORCES       — 0 or 1
#   GPUS_PER_NODE     — >0 enables HIP_VISIBLE_DEVICES pinning; 0 = CPU mode
#   CPUS_PER_SIM      — (informational; NTOMP is derived from SLURM_CPUS_PER_TASK)
#   NTOMP             — explicit thread count if set; otherwise auto-computed
#   NSTEPS_EQ/MIN     — optional MDP overrides

#SBATCH --job-name=lipid-sim
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --mail-type=FAIL
#SBATCH --account=cellmembrane
#SBATCH --output=logs/simulations/submit-%j.out
#SBATCH --error=logs/simulations/submit-%j.err

set -euo pipefail
mkdir -p logs/simulations

cd "$HOME/lipid-graph-nn"

# Source the per-batch env file passed by submit_simulations.sh as $1.
# See sbatch_simulations_general1.sh for rationale; same defensive pattern.
SUBMIT_ENV_FILE=""
if [[ $# -gt 0 && -f "$1" ]]; then
    SUBMIT_ENV_FILE="$1"
    # shellcheck disable=SC1090
    source "$SUBMIT_ENV_FILE"
    echo "Sourced env from: $SUBMIT_ENV_FILE"
    shift
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"

CONDA_ENV=$(python scripts/python/print_config_var.py hpc.conda_env)
conda activate "$CONDA_ENV"

# Load GROMACS (ROCm-enabled build; works on both GPU and CPU-only nodes)
MODULEFILES_PATH=$(python scripts/python/print_config_var.py hpc.modulefiles_path)
MODULE_GROMACS=$(python scripts/python/print_config_var.py hpc.module_gromacs)
module purge
module use "$MODULEFILES_PATH"
module load "$MODULE_GROMACS"

# Validate required env vars
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set by submit_simulations.sh}"
: "${N_SIMS_PER_NODE:?N_SIMS_PER_NODE must be set by submit_simulations.sh}"

N_SIMS="$N_SIMS_PER_NODE"

# ntomp: use explicit value if set, otherwise divide total task CPUs equally
if [[ -n "${NTOMP:-}" ]]; then
    NTOMP_VALUE="$NTOMP"
else
    NTOMP_VALUE=$(( SLURM_CPUS_PER_TASK / N_SIMS ))
fi

echo "Launching $N_SIMS simulation(s) on $(hostname)"
echo "  output root : $OUTPUT_ROOT"
echo "  ntomp/sim   : $NTOMP_VALUE"
echo "  gpu mode    : $([ "${GPUS_PER_NODE:-8}" -gt 0 ] && echo yes || echo no)"
echo ""

PIDS=()
SLOT_COMPS=()

for (( i=0; i<N_SIMS; i++ )); do
    COMP_VAR="RUN_${i}_COMP"
    COMP="${!COMP_VAR:-}"

    if [[ -z "$COMP" ]]; then
        echo "  WARNING: ${COMP_VAR} not set; skipping slot $i" >&2
        PIDS+=(-1)
        SLOT_COMPS+=("")
        continue
    fi

    SLOT_COMPS+=("$COMP")
    OUT_DIR="${OUTPUT_ROOT}/${COMP}"
    mkdir -p "$OUT_DIR"
    LOGOUT="${OUT_DIR}/sim-${SLURM_JOB_ID}-gpu${i}.out"
    LOGERR="${OUT_DIR}/sim-${SLURM_JOB_ID}-gpu${i}.err"

    # Build mdrun extra args: ntomp + optional -nb cpu for CPU-only runs
    MDRUN_EXTRA="-ntomp ${NTOMP_VALUE}"
    [[ "${GPUS_PER_NODE:-8}" -eq 0 ]] && MDRUN_EXTRA+=" -nb cpu"

    # Build run_martini_pipeline.py argument list.  CRITICAL: --mdrun-args
    # uses argparse.REMAINDER, which greedily consumes everything after it.
    # It MUST be the last flag — any --prod-ns / --nsteps / --save-forces
    # placed after --mdrun-args is silently absorbed into mdrun_args and
    # never reaches its own option (real bug observed on general1 first run).
    SIM_ARGS=("$COMP"
        "--out-dir"     "$OUTPUT_ROOT"
        "--maxwarn"     "${MAXWARN:-2}"
    )
    [[ -n "${PROD_NS:-}"    ]] && SIM_ARGS+=("--prod-ns"    "$PROD_NS")
    [[ -n "${NSTEPS:-}"     ]] && SIM_ARGS+=("--nsteps"     "$NSTEPS")
    [[ "${SAVE_FORCES:-0}" -eq 1 ]] && SIM_ARGS+=("--save-forces")
    [[ -n "${NSTEPS_EQ:-}"  ]] && SIM_ARGS+=("--nsteps-eq"  "$NSTEPS_EQ")
    [[ -n "${NSTEPS_MIN:-}" ]] && SIM_ARGS+=("--nsteps-min" "$NSTEPS_MIN")
    # --mdrun-args MUST come LAST (REMAINDER absorbs everything after it)
    SIM_ARGS+=("--mdrun-args" "$MDRUN_EXTRA")

    echo "  [slot $i]  $COMP  → $LOGOUT"

    (
        # GPU pinning: one GPU per slot when GPUS_PER_NODE > 0
        if [[ "${GPUS_PER_NODE:-8}" -gt 0 ]]; then
            export HIP_VISIBLE_DEVICES="$i"
            export CUDA_VISIBLE_DEVICES="$i"
        fi
        export OMP_NUM_THREADS="$NTOMP_VALUE"

        python scripts/simulation/run_martini_pipeline.py "${SIM_ARGS[@]}" \
            >"$LOGOUT" 2>"$LOGERR"
    ) &
    PIDS+=($!)
done

echo ""

# Wait for all slots; surface the worst exit code
EXIT_CODE=0
for (( i=0; i<${#PIDS[@]}; i++ )); do
    PID="${PIDS[$i]}"
    COMP="${SLOT_COMPS[$i]:-"(skipped)"}"

    if [[ "$PID" -eq -1 ]]; then
        echo "  [slot $i] SKIPPED ($COMP)"
        continue
    fi

    if wait "$PID"; then
        echo "  [slot $i] OK  ($COMP)"
    else
        rc=$?
        echo "  [slot $i] FAILED  ($COMP)  exit=$rc"
        EXIT_CODE=$rc
    fi
done

echo ""
echo "All simulations done on $(hostname)  (exit_code=$EXIT_CODE)"
exit "$EXIT_CODE"
