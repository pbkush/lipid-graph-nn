#!/bin/bash
# benchmark_hpc_general1.sh — CPU-only throughput benchmark on Goethe-HLR
# general1 partition (Appendix K / step 10b of martini_pipeline_plan.md).
#
# Two phases (mirrors benchmark_hpc.sh on the GPU side):
#   Phase 1 — TPR setup.  If no usable prun.tpr is available, submit a
#             sbatch_setup_general1.sh job that runs the full pipeline
#             (insane → min → short eq → 1-step prod) with the spack
#             gromacs/2022.4 + openmpi 5.0.5 modules and an mpirun -np 1
#             wrapper around gmx_mpi.
#   Phase 2 — submit one sbatch_benchmark_hpc_general1.sh job per sweep
#             point from benchmark_points_general1.tsv, gated on Phase 1
#             via --dependency=afterok.
#
# TPR handling (per K.12 Q2 answer):
#   default  → build a fresh tpr on general1 with the 2022 toolchain (Phase 1).
#              Outputs land at $TPR_ROOT/POPC100/run/prun.tpr.
#   override → --reference-tpr PATH skips Phase 1 entirely and points at
#              any existing tpr (e.g. one built by the GPU benchmark).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
REFERENCE_TPR=""
NSTEPS=100000
POINTS_FILE="$SCRIPT_DIR/benchmark_points_general1.tsv"
BENCH_ROOT=""
PARTITION="general1"
# 2 h is generous but cheap: SLURM charges actual run time, not requested.
# POPC100 on 40 cores ~ 500-1000 ns/day for single mdrun; ~50-150 ns/day per
# slot when 8 slots share the node.  At 100 ns/day, 100k-step (2 ns) point
# takes ~30 min — exactly the previous default.  Bumping to 2 h leaves
# headroom for memory-bandwidth contention surprises.
TIME_LIMIT="02:00:00"
DRY_RUN=0
SETUP_COMP="POPC100"
SETUP_NSTEPS_EQ=10000      # 200 ps at dt=0.02; enough to relax forces for benchmarking
SETUP_NSTEPS_MIN=1000
SETUP_NSTEPS_PROD=1        # we only need prun.tpr to exist; 1-step prod is fine
SETUP_TIME="01:00:00"      # generous; CPU pipeline run finishes in a few min

usage() {
    cat >&2 <<'EOF'
usage: benchmark_hpc_general1.sh
    [--reference-tpr PATH]    skip Phase 1 and use this tpr directly
    [--setup-comp NAME]       composition to build the tpr from   (default: POPC100)
    [--setup-nsteps-eq N]     eq steps in Phase 1 setup           (default: 10000)
    [--setup-nsteps-min N]    minimization steps                  (default: 1000)
    [--setup-time HH:MM:SS]   wall time for the Phase 1 job       (default: 01:00:00)
    [--nsteps N]              mdrun -nsteps per benchmark slot    (default: 100000)
    [--points-file PATH]      sweep table                         (default: benchmark_points_general1.tsv)
    [--bench-root PATH]       results dir
                              (default: results/benchmarks/martini_pipeline/<date>/general1)
    [--partition NAME]        partition                           (default: general1)
    [--time HH:MM:SS]         wall time per benchmark point       (default: 02:00:00)
    [--dry-run]               print sbatch commands, do not submit
EOF
    exit 1
}

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference-tpr)    REFERENCE_TPR="$2";     shift 2 ;;
        --setup-comp)       SETUP_COMP="$2";        shift 2 ;;
        --setup-nsteps-eq)  SETUP_NSTEPS_EQ="$2";   shift 2 ;;
        --setup-nsteps-min) SETUP_NSTEPS_MIN="$2";  shift 2 ;;
        --setup-time)       SETUP_TIME="$2";        shift 2 ;;
        --nsteps)           NSTEPS="$2";            shift 2 ;;
        --points-file)      POINTS_FILE="$2";       shift 2 ;;
        --bench-root)       BENCH_ROOT="$2";        shift 2 ;;
        --partition)        PARTITION="$2";         shift 2 ;;
        --time)             TIME_LIMIT="$2";        shift 2 ;;
        --dry-run)          DRY_RUN=1;              shift ;;
        --help|-h)          usage ;;
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
# TPR root for Phase-1-built tprs (separate from the GPU benchmark's root, so
# the two toolchains don't trample each other).
GENERAL1_TPR_ROOT="/work/${GROUP}/${USER}/${WORK_SUBPATH}/${HPC_OUTPUT_SUBPATH}/general1_benchmark"

# Resolve reference tpr: --reference-tpr wins; else default to Phase-1 output path.
SKIP_PHASE_1=0
if [[ -n "$REFERENCE_TPR" ]]; then
    SKIP_PHASE_1=1
else
    REFERENCE_TPR="${GENERAL1_TPR_ROOT}/${SETUP_COMP}/run/prun.tpr"
fi

printf '\nCPU benchmark on general1  (%s)\n' "$(date '+%Y-%m-%d %H:%M')"
printf '  reference tpr    : %s\n' "$REFERENCE_TPR"
if [[ "$SKIP_PHASE_1" -eq 1 ]]; then
    printf '  phase 1 setup    : SKIPPED (--reference-tpr provided)\n'
else
    printf '  phase 1 setup    : %s (eq=%s, min=%s, prod=%s)\n' \
        "$SETUP_COMP" "$SETUP_NSTEPS_EQ" "$SETUP_NSTEPS_MIN" "$SETUP_NSTEPS_PROD"
fi
printf '  nsteps/slot      : %s\n' "$NSTEPS"
printf '  partition        : %s\n' "$PARTITION"
printf '  bench root       : %s\n' "$BENCH_ROOT"
printf '\n'

[[ ! -f "$POINTS_FILE" ]] && { echo "ERROR: points file not found: $POINTS_FILE" >&2; exit 1; }

mkdir -p "$BENCH_ROOT" logs/benchmarks

# ── Phase 1: build the tpr on general1 with the 2022 toolchain ────────────────
SETUP_JOB_ID=""
if [[ "$SKIP_PHASE_1" -eq 0 ]]; then
    # Skip phase 1 if the tpr already exists (idempotent re-run).
    if [[ "$DRY_RUN" -eq 0 && -f "$REFERENCE_TPR" ]]; then
        printf '  [setup] %s: prun.tpr already at %s — skipping Phase 1\n' \
            "$SETUP_COMP" "$REFERENCE_TPR"
    else
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '  [DRY RUN] COMP=%s OUTPUT_ROOT=%s NSTEPS_EQ=%s NSTEPS_MIN=%s NSTEPS_PROD=%s NTOMP=40 ' \
                "$SETUP_COMP" "$GENERAL1_TPR_ROOT" "$SETUP_NSTEPS_EQ" \
                "$SETUP_NSTEPS_MIN" "$SETUP_NSTEPS_PROD"
            printf 'sbatch --partition=%s --time=%s --cpus-per-task=40 --mem=64G --export=ALL %s\n' \
                "$PARTITION" "$SETUP_TIME" "$SCRIPT_DIR/sbatch_setup_general1.sh"
            printf '    [setup] would submit Phase 1 job\n'
        else
            printf '  [setup] %s: prun.tpr missing — submitting Phase 1 setup job\n' "$SETUP_COMP"
            SETUP_JOB_ID=$(
                COMP="$SETUP_COMP" \
                OUTPUT_ROOT="$GENERAL1_TPR_ROOT" \
                NSTEPS_EQ="$SETUP_NSTEPS_EQ" \
                NSTEPS_MIN="$SETUP_NSTEPS_MIN" \
                NSTEPS_PROD="$SETUP_NSTEPS_PROD" \
                NTOMP=40 \
                MAXWARN=2 \
                sbatch \
                    --partition="$PARTITION" \
                    --time="$SETUP_TIME" \
                    --cpus-per-task=40 \
                    --mem=64G \
                    --export=ALL \
                    "$SCRIPT_DIR/sbatch_setup_general1.sh" \
                | awk '{print $NF}'
            )
            printf '  [setup] %s: job %s submitted\n' "$SETUP_COMP" "$SETUP_JOB_ID"
        fi
    fi
fi
printf '\n'

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
        # Compose dependency: afterok on Phase 1 setup job (so bench points
        # don't run against a missing/incomplete tpr) AND afterany on the
        # previous bench job (so SLURM only runs one bench point at a time).
        DEP_PARTS=()
        [[ -n "$SETUP_JOB_ID"      ]] && DEP_PARTS+=("afterok:${SETUP_JOB_ID}")
        [[ -n "$PREV_BENCH_JOB_ID" ]] && DEP_PARTS+=("afterany:${PREV_BENCH_JOB_ID}")
        DEP_ARG=""
        if [[ ${#DEP_PARTS[@]} -gt 0 ]]; then
            IFS=, ; DEP_ARG="--dependency=${DEP_PARTS[*]}" ; IFS=$' \t\n'
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
