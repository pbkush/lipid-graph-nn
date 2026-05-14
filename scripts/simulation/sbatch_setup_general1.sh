#!/bin/bash
# sbatch_setup_general1.sh — Phase 1 worker for the general1 CPU benchmark.
#
# Runs the full martini pipeline (insane → minimization → equilibration →
# short production) on the general1 partition to mint a valid prun.tpr that
# the Phase 2 benchmark points can consume.  Uses GROMACS 2022 + openmpi via
# the _gmx_mpi_wrapper.sh shim, so pipeline.run()'s single `gmx_executable`
# parameter works unchanged.
#
# Env vars set by benchmark_hpc_general1.sh:
#   COMP             — composition name (e.g. POPC100)
#   OUTPUT_ROOT      — output root (e.g. /work/.../general1_benchmark)
#   NSTEPS_PROD      — production steps (default: 1; just need the tpr to exist)
#   NSTEPS_EQ        — equilibration steps (default: 10000 = 200 ps at dt=0.02)
#   NSTEPS_MIN       — minimization steps (default: 1000)
#   NTOMP            — OpenMP threads for mdrun (default: 40 = full general1 node)
#   MAXWARN          — grompp -maxwarn (default: 2)
#
# A short eq (200 ps) is enough to relax insane's initial coords into a state
# whose forces no longer blow up step 0 of the benchmark.  The benchmark
# measures mdrun perf, not physical observables — so full eq is not required.

#SBATCH --job-name=lipid-bench-setup
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --mail-type=FAIL
#SBATCH --account=cellmembrane
#SBATCH --output=logs/benchmarks/setup-cpu-%j.out
#SBATCH --error=logs/benchmarks/setup-cpu-%j.err

set -euo pipefail
mkdir -p logs/benchmarks

cd "$HOME/lipid-graph-nn"

# Conda env (Python pipeline)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV=$(python scripts/python/print_config_var.py hpc.conda_env)
conda activate "$CONDA_ENV"

# general1 toolchain
module purge
module load mpi/openmpi/5.0.5-rocm
module load gromacs/2022.4-gcc-11.3.1-zx2wwcx

: "${COMP:?COMP must be set by benchmark_hpc_general1.sh}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be set by benchmark_hpc_general1.sh}"

NSTEPS_PROD="${NSTEPS_PROD:-1}"
NSTEPS_EQ="${NSTEPS_EQ:-10000}"
NSTEPS_MIN="${NSTEPS_MIN:-1000}"
NTOMP="${NTOMP:-40}"
MAXWARN="${MAXWARN:-2}"

echo "general1 setup on $(hostname)"
echo "  composition  : $COMP"
echo "  output root  : $OUTPUT_ROOT"
echo "  nsteps min   : $NSTEPS_MIN"
echo "  nsteps eq    : $NSTEPS_EQ"
echo "  nsteps prod  : $NSTEPS_PROD"
echo "  ntomp        : $NTOMP"
echo ""

# Thread pinning (matches the benchmark worker's environment for consistency)
export OMP_NUM_THREADS="$NTOMP"
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Run the pipeline.  -nb cpu disables GPU offload (no GPUs here); -ntomp uses
# all the cores we asked SLURM for.
python scripts/simulation/run_martini_pipeline.py "$COMP" \
    --output-root "$OUTPUT_ROOT" \
    --gmx        "$PWD/scripts/simulation/_gmx_mpi_wrapper.sh" \
    --nsteps     "$NSTEPS_PROD" \
    --nsteps-eq  "$NSTEPS_EQ" \
    --nsteps-min "$NSTEPS_MIN" \
    --maxwarn    "$MAXWARN" \
    --mdrun-args "-ntomp $NTOMP -nb cpu"

echo ""
echo "general1 setup done on $(hostname)  (composition=$COMP)"
echo "tpr at: $OUTPUT_ROOT/$COMP/run/prun.tpr"
