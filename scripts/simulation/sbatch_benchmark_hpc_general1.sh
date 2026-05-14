#!/bin/bash
# sbatch_benchmark_hpc_general1.sh — SLURM worker for one CPU benchmark sweep point.
#
# Runs SIMS_PER_NODE parallel `mpirun -np MPI_RANKS_PER_SIM gmx_mpi mdrun` slots
# on a single general1 node, using a pre-built prun.tpr.  Mirrors the GPU
# worker (sbatch_benchmark_hpc.sh) but loads the spack GROMACS 2022 + openmpi
# modules and uses gmx_mpi via mpirun for both single- and multi-rank slots.
#
# Env vars set by benchmark_hpc_general1.sh at submission time:
#   BENCH_POINT_DIR     — directory to write slot logs
#   REFERENCE_TPRS      — colon-separated tpr paths (cycled across slots)
#   SIMS_PER_NODE       — number of parallel slots
#   MPI_RANKS_PER_SIM   — mpirun -np value per slot
#   CPUS_PER_SIM        — OpenMP threads per slot (gmx_mpi mdrun -ntomp)
#   NSTEPS              — gmx mdrun -nsteps

#SBATCH --job-name=lipid-bench-cpu
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --mail-type=FAIL
#SBATCH --account=cellmembrane
#SBATCH --output=logs/benchmarks/bench-cpu-%j.out
#SBATCH --error=logs/benchmarks/bench-cpu-%j.err

set -euo pipefail
mkdir -p logs/benchmarks

cd "$HOME/lipid-graph-nn"

# spack GROMACS 2022 + openmpi (general1 toolchain, per K.12 Q & user-supplied
# module commands).  No conda env needed — analysis runs on the login node.
module purge
module load mpi/openmpi/5.0.0
module load gromacs/2022.4-gcc-11.3.1-zx2wwcx

: "${BENCH_POINT_DIR:?BENCH_POINT_DIR must be set by benchmark_hpc_general1.sh}"
: "${REFERENCE_TPRS:?REFERENCE_TPRS must be set by benchmark_hpc_general1.sh}"
: "${SIMS_PER_NODE:?SIMS_PER_NODE must be set by benchmark_hpc_general1.sh}"
: "${MPI_RANKS_PER_SIM:?MPI_RANKS_PER_SIM must be set by benchmark_hpc_general1.sh}"
: "${CPUS_PER_SIM:?CPUS_PER_SIM must be set by benchmark_hpc_general1.sh}"

N_SIMS="$SIMS_PER_NODE"
N_RANKS="$MPI_RANKS_PER_SIM"
NTOMP="$CPUS_PER_SIM"
NSTEPS="${NSTEPS:-100000}"

mkdir -p "$BENCH_POINT_DIR"

IFS=: read -ra TPRS_ARR <<< "$REFERENCE_TPRS"
N_TPRS=${#TPRS_ARR[@]}

echo "CPU benchmark point on $(hostname)"
echo "  point dir       : $BENCH_POINT_DIR"
echo "  sims            : $N_SIMS"
echo "  mpi_ranks/sim   : $N_RANKS"
echo "  ntomp/sim       : $NTOMP"
echo "  nsteps          : $NSTEPS"
echo ""

PIDS=()

for (( i=0; i<N_SIMS; i++ )); do
    TPR="${TPRS_ARR[$(( i % N_TPRS ))]}"
    SLOT_DIR="$BENCH_POINT_DIR/slot_${i}"
    mkdir -p "$SLOT_DIR"

    # No -resethway on CPU (K.2 Decision 3 / K.12 Q4): GROMACS 2022 PME tuning
    # behaviour differs from v2025.4, and we want clean step-0 throughput so
    # the GPU/CPU comparison isn't biased by version-specific warm-up.
    #
    # --map-by :OVERSUBSCRIBE: SLURM allocates --cpus-per-task=40 but only
    # --ntasks=1 (the default), so openmpi/PRRTE sees 1 "slot" and refuses
    # to launch -np 4.  Telling openmpi to oversubscribe tells it we manage
    # the rank-to-core mapping via SLURM + OMP_NUM_THREADS; the kernel
    # scheduler still respects our cgroup-bounded core count.
    #
    # No -nstxout/-nstvout/etc.: those are MDP options, not mdrun CLI flags
    # in GROMACS 2022.  Output frequency is baked into the TPR from prun.mdp.
    MDRUN_ARGS=(
        mpirun
        --map-by ":OVERSUBSCRIBE"
        -np "$N_RANKS"
        gmx_mpi mdrun
        -s    "$TPR"
        -deffnm bench
        -nsteps "$NSTEPS"
        -ntomp  "$NTOMP"
        -nb cpu
    )

    echo "  [slot $i]  $(basename "$(dirname "$TPR")")  → $SLOT_DIR/bench.log"

    (
        # Thread pinning (K.12 Q5: on).
        export OMP_NUM_THREADS="$NTOMP"
        export OMP_PLACES=cores
        export OMP_PROC_BIND=close
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

echo ""
echo "CPU benchmark point done on $(hostname)  (exit_code=$EXIT_CODE)"
exit "$EXIT_CODE"
