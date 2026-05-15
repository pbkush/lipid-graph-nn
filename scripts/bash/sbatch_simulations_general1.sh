#!/bin/bash
# sbatch_simulations_general1.sh — SLURM worker for Martini 3 bilayer
# simulations on the general1 CPU partition (Goethe-HLR, spack GROMACS 2022).
#
# CPU equivalent of sbatch_simulations.sh.  Loads the openmpi + GROMACS-2022
# modules, runs the pipeline through the _gmx_mpi_wrapper.sh shim (so the
# pipeline's single `gmx_executable` parameter works on a partition where
# only `gmx_mpi` is installed), and uses OpenMP thread pinning.
#
# Env vars set by submit_simulations.sh at submission time:
#   OUTPUT_ROOT         — top-level output directory (on /work)
#   N_SIMS_PER_NODE     — number of parallel simulations to run on this node
#   MPI_RANKS_PER_SIM   — mpirun -np per slot (1 in production by default;
#                          inherited by the wrapper)
#   CPUS_PER_SIM        — OpenMP threads per slot (gmx_mpi mdrun -ntomp)
#   RUN_<i>_COMP        — canonical composition name for slot i
#   PROD_NS or NSTEPS   — production length
#   MAXWARN             — gmx grompp -maxwarn value
#   SAVE_FORCES         — 0 or 1
#   NSTEPS_EQ/MIN       — optional MDP overrides

#SBATCH --job-name=lipid-sim-cpu
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --mail-type=FAIL
#SBATCH --account=cellmembrane
#SBATCH --output=logs/simulations/sim-cpu-%j.out
#SBATCH --error=logs/simulations/sim-cpu-%j.err

set -euo pipefail
mkdir -p logs/simulations

cd "$HOME/lipid-graph-nn"

# Source the per-batch env file passed by submit_simulations.sh as $1.
# Avoids SLURM --export-propagation quirks: positional args always reach
# the worker process verbatim.  Backward-compat: if no arg is passed or
# the file doesn't exist, fall through to using whatever env vars are
# already set (legacy path / manual invocation).
SUBMIT_ENV_FILE=""
if [[ $# -gt 0 && -f "$1" ]]; then
    SUBMIT_ENV_FILE="$1"
    # shellcheck disable=SC1090  # dynamic path is the whole point
    source "$SUBMIT_ENV_FILE"
    echo "Sourced env from: $SUBMIT_ENV_FILE"
    shift
fi
echo "[diag] after source-env-file: PROD_NS='${PROD_NS:-<unset>}'  NSTEPS='${NSTEPS:-<unset>}'  OUTPUT_ROOT='${OUTPUT_ROOT:-<unset>}'"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
echo "[diag] after source-conda.sh: PROD_NS='${PROD_NS:-<unset>}'"

CONDA_ENV=$(python scripts/python/print_config_var.py hpc.conda_env)
conda activate "$CONDA_ENV"
echo "[diag] after conda activate:   PROD_NS='${PROD_NS:-<unset>}'"

# general1 toolchain: spack openmpi + GROMACS-2022.  Module names come from
# the hpc_defaults_cpu block in config.yaml so they're version-locked alongside
# the calibrated benchmark numbers.
MODULE_MPI=$(python scripts/python/print_config_var.py \
    martini_pipeline.hpc_defaults_cpu.module_mpi_cpu)
MODULE_GROMACS_CPU=$(python scripts/python/print_config_var.py \
    martini_pipeline.hpc_defaults_cpu.module_gromacs_cpu)
module purge
echo "[diag] after module purge:     PROD_NS='${PROD_NS:-<unset>}'"
module load "$MODULE_MPI"
module load "$MODULE_GROMACS_CPU"
echo "[diag] after module load:      PROD_NS='${PROD_NS:-<unset>}'"

# Defensive: re-source the env file AFTER conda+module, so anything they
# unset (e.g. a stray `unset PROD_NS` in a conda activate.d script) is
# restored before the slot loop reads PROD_NS.
if [[ -n "$SUBMIT_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SUBMIT_ENV_FILE"
    echo "[diag] after re-source:        PROD_NS='${PROD_NS:-<unset>}'"
fi

# Validate required env vars
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set by submit_simulations.sh}"
: "${N_SIMS_PER_NODE:?N_SIMS_PER_NODE must be set by submit_simulations.sh}"
: "${CPUS_PER_SIM:?CPUS_PER_SIM must be set by submit_simulations.sh}"
: "${MPI_RANKS_PER_SIM:?MPI_RANKS_PER_SIM must be set by submit_simulations.sh}"

N_SIMS="$N_SIMS_PER_NODE"

# OMP thread count = CPUS_PER_SIM (Decision 60 / L.4 step 5).  Unlike the GPU
# worker we DON'T divide SLURM_CPUS_PER_TASK by N_SIMS — the orchestrator
# already requested sims × ranks × cpus_per_sim total CPUs, so the per-slot
# carve-out is exactly cpus_per_sim per (sim, rank).
NTOMP_VALUE="$CPUS_PER_SIM"

echo "Launching $N_SIMS simulation(s) on $(hostname)  [general1 CPU]"
echo "  output root      : $OUTPUT_ROOT"
echo "  ntomp/sim        : $NTOMP_VALUE"
echo "  mpi_ranks/sim    : $MPI_RANKS_PER_SIM"
echo "  module (gromacs) : $MODULE_GROMACS_CPU"
echo "  module (mpi)     : $MODULE_MPI"
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
    LOGOUT="${OUT_DIR}/sim-${SLURM_JOB_ID}-cpu${i}.out"
    LOGERR="${OUT_DIR}/sim-${SLURM_JOB_ID}-cpu${i}.err"

    # mdrun extra args: -ntomp + explicit -nb cpu (no GPU on general1)
    MDRUN_EXTRA="-ntomp ${NTOMP_VALUE} -nb cpu"

    # Build run_martini_pipeline.py argument list.  --gmx points at the
    # mpirun-wrapped gmx_mpi shim (Decision 58 / step 10b).
    echo "[diag] slot $i: PROD_NS='${PROD_NS:-<unset>}'  NSTEPS='${NSTEPS:-<unset>}'  about to build SIM_ARGS"

    SIM_ARGS=("$COMP"
        "--out-dir"    "$OUTPUT_ROOT"
        "--gmx"        "$PWD/scripts/simulation/_gmx_mpi_wrapper.sh"
        "--maxwarn"    "${MAXWARN:-2}"
        "--mdrun-args" "$MDRUN_EXTRA"
    )
    [[ -n "${PROD_NS:-}"    ]] && SIM_ARGS+=("--prod-ns"    "$PROD_NS")
    [[ -n "${NSTEPS:-}"     ]] && SIM_ARGS+=("--nsteps"     "$NSTEPS")
    [[ "${SAVE_FORCES:-0}" -eq 1 ]] && SIM_ARGS+=("--save-forces")
    [[ -n "${NSTEPS_EQ:-}"  ]] && SIM_ARGS+=("--nsteps-eq"  "$NSTEPS_EQ")
    [[ -n "${NSTEPS_MIN:-}" ]] && SIM_ARGS+=("--nsteps-min" "$NSTEPS_MIN")

    # Print the FULL SIM_ARGS list verbatim so we can see exactly what python
    # will be invoked with.  One arg per line for unambiguous inspection.
    echo "[diag] slot $i: SIM_ARGS has ${#SIM_ARGS[@]} elements:"
    for arg in "${SIM_ARGS[@]}"; do
        printf '[diag]   |%s|\n' "$arg"
    done

    echo "  [slot $i]  $COMP  → $LOGOUT"

    (
        # CPU thread pinning (Decision 56): avoid OS thread migration on a
        # shared general1 node so per-slot throughput is reproducible.
        export OMP_NUM_THREADS="$NTOMP_VALUE"
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close

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
echo "All simulations done on $(hostname)  [general1]  (exit_code=$EXIT_CODE)"
exit "$EXIT_CODE"
