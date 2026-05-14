#!/bin/bash
# benchmark_hpc_general1.sh — CPU-only throughput benchmark on Goethe-HLR
# general1 partition (Appendix K / step 10b of martini_pipeline_plan.md).
#
# Submits one sbatch_benchmark_hpc_general1.sh job per sweep point from
# benchmark_points_general1.tsv.  Uses the GROMACS-2022 + openmpi modules:
#     module load mpi/openmpi/5.0.5-rocm
#     module load gromacs/2022.4-gcc-11.3.1-zx2wwcx
# (loaded inside the worker, not here.)
#
# TPR handling (per K.12 Q2 answer):
#   default  → reuse the v2025.4-built prun.tpr at
#              /work/.../martini_pipeline/benchmark/POPC100/run/prun.tpr
#              (produced by Phase 1 of benchmark_hpc.sh).  This is forward-
#              compatible — gmx_mpi 2022 reads v2025-built TPRs with a
#              "downgrade" warning but identical integration.
#   override → --reference-tpr PATH to point at any other tpr
#
# A self-contained Phase 1 (build tpr on general1 with the 2022 toolchain)
# is deferred — see K.12 Q2(b) and the "future work" note at the bottom.
# For now the script fails fast if neither default nor --reference-tpr is
# available, with a clear hint to run benchmark_hpc.sh first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
REFERENCE_TPR=""
NSTEPS=100000
POINTS_FILE="$SCRIPT_DIR/benchmark_points_general1.tsv"
BENCH_ROOT=""
PARTITION="general1"
TIME_LIMIT="00:30:00"
DRY_RUN=0

usage() {
    cat >&2 <<'EOF'
usage: benchmark_hpc_general1.sh
    [--reference-tpr PATH]   override the default v2025.4 tpr location
    [--nsteps N]             mdrun -nsteps per slot   (default: 100000)
    [--points-file PATH]     sweep table              (default: benchmark_points_general1.tsv)
    [--bench-root PATH]      results dir
                             (default: results/benchmarks/martini_pipeline/<date>/general1)
    [--partition NAME]       partition                (default: general1)
    [--time HH:MM:SS]        wall time per point      (default: 00:30:00)
    [--dry-run]              print sbatch commands, do not submit
EOF
    exit 1
}

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference-tpr) REFERENCE_TPR="$2"; shift 2 ;;
        --nsteps)        NSTEPS="$2";        shift 2 ;;
        --points-file)   POINTS_FILE="$2";   shift 2 ;;
        --bench-root)    BENCH_ROOT="$2";    shift 2 ;;
        --partition)     PARTITION="$2";     shift 2 ;;
        --time)          TIME_LIMIT="$2";    shift 2 ;;
        --dry-run)       DRY_RUN=1;          shift ;;
        --help|-h)       usage ;;
        *) echo "ERROR: unknown option: $1" >&2; usage ;;
    esac
done

# ── config ────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
_cfg() { python scripts/python/print_config_var.py "$1"; }
GROUP="${GROUP:-$(_cfg hpc.group)}"
WORK_SUBPATH=$(_cfg hpc.work_subpath)
HPC_OUTPUT_SUBPATH=$(_cfg martini_pipeline.hpc_output_subpath)

# ── paths ─────────────────────────────────────────────────────────────────────
BENCH_DATE=$(date +%Y-%m-%d)
if [[ -z "$BENCH_ROOT" ]]; then
    BENCH_ROOT="$REPO_ROOT/results/benchmarks/martini_pipeline/$BENCH_DATE/general1"
fi
DEFAULT_TPR_ROOT="/work/${GROUP}/${USER}/${WORK_SUBPATH}/${HPC_OUTPUT_SUBPATH}/benchmark"
if [[ -z "$REFERENCE_TPR" ]]; then
    REFERENCE_TPR="${DEFAULT_TPR_ROOT}/POPC100/run/prun.tpr"
fi

printf '\nCPU benchmark on general1  (%s)\n' "$(date '+%Y-%m-%d %H:%M')"
printf '  reference tpr    : %s\n' "$REFERENCE_TPR"
printf '  nsteps/slot      : %s\n' "$NSTEPS"
printf '  partition        : %s\n' "$PARTITION"
printf '  bench root       : %s\n' "$BENCH_ROOT"
printf '\n'

# ── existence check (skip in --dry-run; tpr may not exist locally) ────────────
if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -f "$REFERENCE_TPR" ]]; then
        cat >&2 <<EOF
ERROR: reference TPR not found at:
  $REFERENCE_TPR

Either:
  1. Run the GPU benchmark first to build it:
       bash scripts/simulation/benchmark_hpc.sh
  2. Pass an existing TPR via --reference-tpr PATH

(Building a TPR on general1 with the 2022 toolchain is deferred — see K.12 Q2
in docs/martini_pipeline_plan.md.)
EOF
        exit 1
    fi
fi

[[ ! -f "$POINTS_FILE" ]] && { echo "ERROR: points file not found: $POINTS_FILE" >&2; exit 1; }

mkdir -p "$BENCH_ROOT" logs/benchmarks

# ── Phase 2: submit one sbatch per sweep point ────────────────────────────────
BATCH_COUNT=0
PREV_BENCH_JOB_ID=""   # for afterany chaining (minimises concurrent pending bench jobs)

while IFS=$'\t' read -r LABEL SIMS RANKS CPUS MEM POINT_PARTITION REST; do
    [[ -z "$LABEL" || "$LABEL" == \#* ]] && continue

    EFFECTIVE_PARTITION="$PARTITION"
    TOTAL_CPUS=$(( SIMS * RANKS * CPUS ))
    MEM_NUM="${MEM%[A-Za-z]*}"
    MEM_UNIT="${MEM##*[0-9]}"
    TOTAL_MEM="$(( MEM_NUM * SIMS ))${MEM_UNIT}"

    POINT_DIR="$BENCH_ROOT/points/$LABEL"
    mkdir -p "$POINT_DIR"

    # Write point_meta.json so analyze_benchmark.py can attribute logs back.
    python3 -c "
import json, sys
meta = {
    'label': sys.argv[1],
    'sims_per_node': int(sys.argv[2]),
    'mpi_ranks_per_sim': int(sys.argv[3]),
    'cpus_per_sim': int(sys.argv[4]),
    'gpus_per_node': 0,
    'mem_per_sim': sys.argv[5],
    'partition': sys.argv[6],
    'device': 'cpu',
}
with open(sys.argv[7], 'w') as f:
    json.dump(meta, f, indent=2)
" "$LABEL" "$SIMS" "$RANKS" "$CPUS" "$MEM" "$EFFECTIVE_PARTITION" \
  "$POINT_DIR/point_meta.json"

    BATCH_COUNT=$(( BATCH_COUNT + 1 ))

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '  [DRY RUN] BENCH_POINT_DIR=%s REFERENCE_TPRS=%s SIMS_PER_NODE=%s MPI_RANKS_PER_SIM=%s CPUS_PER_SIM=%s NSTEPS=%s ' \
            "$POINT_DIR" "$REFERENCE_TPR" "$SIMS" "$RANKS" "$CPUS" "$NSTEPS"
        printf 'sbatch --partition=%s --time=%s --cpus-per-task=%s --mem=%s --export=ALL %s\n' \
            "$EFFECTIVE_PARTITION" "$TIME_LIMIT" "$TOTAL_CPUS" "$TOTAL_MEM" \
            "$SCRIPT_DIR/sbatch_benchmark_hpc_general1.sh"
        printf '    label=%s  sims=%s  mpi_ranks/sim=%s  cpus/sim=%s  total_cpus=%s  mem=%s\n' \
            "$LABEL" "$SIMS" "$RANKS" "$CPUS" "$TOTAL_CPUS" "$TOTAL_MEM"
    else
        # afterany chain (no Phase 1 here — TPR is already built or provided).
        DEP_ARG=""
        if [[ -n "$PREV_BENCH_JOB_ID" ]]; then
            DEP_ARG="--dependency=afterany:${PREV_BENCH_JOB_ID}"
        fi

        JOB_ID=$(
            BENCH_POINT_DIR="$POINT_DIR" \
            REFERENCE_TPRS="$REFERENCE_TPR" \
            SIMS_PER_NODE="$SIMS" \
            MPI_RANKS_PER_SIM="$RANKS" \
            CPUS_PER_SIM="$CPUS" \
            NSTEPS="$NSTEPS" \
            sbatch \
                --partition="$EFFECTIVE_PARTITION" \
                --time="$TIME_LIMIT" \
                --cpus-per-task="$TOTAL_CPUS" \
                --mem="$TOTAL_MEM" \
                ${DEP_ARG:+"$DEP_ARG"} \
                --export=ALL \
                "$SCRIPT_DIR/sbatch_benchmark_hpc_general1.sh" \
            | awk '{print $NF}'
        )
        PREV_BENCH_JOB_ID="$JOB_ID"
        printf '  [bench]  %s: job %s  (sims=%s mpi=%s ntomp=%s total=%s cpus)\n' \
            "$LABEL" "$JOB_ID" "$SIMS" "$RANKS" "$CPUS" "$TOTAL_CPUS"
    fi

done < "$POINTS_FILE"

printf '\n%d CPU benchmark job(s) queued\n' "$BATCH_COUNT"
printf 'After all jobs complete, run:\n'
printf '  python scripts/python/analyze_benchmark.py --root %s --recommend --cpu\n' \
    "$(dirname "$BENCH_ROOT")"
