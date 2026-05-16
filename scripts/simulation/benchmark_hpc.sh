#!/bin/bash
# benchmark_hpc.sh — orchestrate HPC throughput benchmark for martini_pipeline.
#
# Phase 1 (setup): for each reference composition missing a prun.tpr, submit a
#   full pipeline run (insane + min + eq + short production) via sbatch_simulations.sh.
#   This also validates the step-9 submission layer end-to-end on HPC.
# Phase 2 (benchmark): submit one sbatch_benchmark_hpc.sh job per sweep point
#   from benchmark_points.tsv, chained after Phase 1 via --dependency=afterok.
# Phase 3 (manual): after jobs complete, run analyze_benchmark.py.
#
# --dry-run prints sbatch commands without submitting; Phase 1 check is skipped
# entirely in dry-run mode (TPRs may not exist locally).
#
# Reference compositions:
#   --reference-comp POPC100           → single-system benchmark
#   --reference-comp POPC100 DPPC100 CHOL40_DPPC60  → 3-system average
# Slots cycle through the reference comp list to fill sims_per_node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
declare -a REFERENCE_COMPS=()
NSTEPS=100000
SETUP_NSTEPS=1000
POINTS_FILE="$SCRIPT_DIR/benchmark_points.tsv"
BENCH_ROOT=""
PARTITION="gpu_test"
SETUP_PARTITION=""
TIME_LIMIT="00:30:00"
DRY_RUN=0

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat >&2 <<'EOF'
usage: benchmark_hpc.sh
    [--reference-comp COMP [COMP ...]]   default: POPC100
    [--nsteps N]                          mdrun steps/slot for benchmark  (default: 100000)
    [--setup-nsteps N]                    mdrun steps for Phase 1 tpr-build (default: 1000)
    [--points-file PATH]                  sweep table  (default: benchmark_points.tsv)
    [--bench-root PATH]                   results output dir
                                          (default: results/benchmarks/martini_pipeline/<date>)
    [--partition NAME]                    GPU partition  (default: gpu_test)
    [--setup-partition NAME]              partition for Phase 1  (default: same as --partition)
    [--time HH:MM:SS]                     wall time per benchmark job  (default: 00:30:00)
    [--dry-run]                           print sbatch commands, do not submit
EOF
    exit 1
}

# ── arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reference-comp)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                REFERENCE_COMPS+=("$1"); shift
            done
            ;;
        --nsteps)           NSTEPS="$2";          shift 2 ;;
        --setup-nsteps)     SETUP_NSTEPS="$2";    shift 2 ;;
        --points-file)      POINTS_FILE="$2";     shift 2 ;;
        --bench-root)       BENCH_ROOT="$2";      shift 2 ;;
        --partition)        PARTITION="$2";       shift 2 ;;
        --setup-partition)  SETUP_PARTITION="$2"; shift 2 ;;
        --time)             TIME_LIMIT="$2";      shift 2 ;;
        --dry-run)          DRY_RUN=1;            shift ;;
        --help|-h)          usage ;;
        *) echo "ERROR: unknown option: $1" >&2; usage ;;
    esac
done

[[ ${#REFERENCE_COMPS[@]} -eq 0 ]] && REFERENCE_COMPS=("POPC100")
SETUP_PARTITION="${SETUP_PARTITION:-$PARTITION}"

# ── config ────────────────────────────────────────────────────────────────────
cd "$REPO_ROOT"
_cfg() { python scripts/python/print_config_var.py "$1"; }
GROUP="${GROUP:-$(_cfg hpc.group)}"
WORK_SUBPATH=$(_cfg hpc.work_subpath)
HPC_OUTPUT_SUBPATH=$(_cfg martini_pipeline.hpc_output_subpath)

# ── paths ─────────────────────────────────────────────────────────────────────
BENCH_DATE=$(date +%Y-%m-%d)
if [[ -z "$BENCH_ROOT" ]]; then
    BENCH_ROOT="$REPO_ROOT/results/benchmarks/martini_pipeline/$BENCH_DATE"
fi
BENCH_SIM_ROOT="/work/${GROUP}/${USER}/${WORK_SUBPATH}/${HPC_OUTPUT_SUBPATH}/benchmark"

printf '\nBenchmark HPC  (%s)\n' "$(date '+%Y-%m-%d %H:%M')"
printf '  reference comps  : %s\n' "${REFERENCE_COMPS[*]}"
printf '  nsteps/slot      : %s\n' "$NSTEPS"
printf '  partition        : %s\n' "$PARTITION"
printf '  bench root       : %s\n' "$BENCH_ROOT"
if [[ "$DRY_RUN" -eq 0 ]]; then
    printf '  sim root         : %s\n' "$BENCH_SIM_ROOT"
fi
printf '\n'

[[ "$DRY_RUN" -eq 1 ]] && printf '  NOTE: --dry-run; Phase 1 TPR check skipped\n\n'

mkdir -p "$BENCH_ROOT"

# ── Phase 1: ensure prun.tpr exists for each reference comp ───────────────────
declare -a SETUP_JOB_IDS=()
declare -a REFERENCE_TPRS=()

for COMP in "${REFERENCE_COMPS[@]}"; do
    TPR="${BENCH_SIM_ROOT}/${COMP}/run/prun.tpr"
    REFERENCE_TPRS+=("$TPR")

    if [[ "$DRY_RUN" -eq 1 ]]; then
        continue
    fi

    if [[ -f "$TPR" ]]; then
        printf '  [setup] %s: prun.tpr exists\n' "$COMP"
        continue
    fi

    printf '  [setup] %s: prun.tpr missing — submitting pipeline setup job\n' "$COMP"

    # Bulletproof export: set the variables as a command-prefix assignment so
    # they live only for this `sbatch` invocation, then use --export=ALL to
    # have SLURM inherit them.  Avoids the SLURM-version-specific parsing
    # quirks around `--export=ALL,VAR=val` (silently dropped entries on
    # Goethe-HLR's SLURM build) and doesn't pollute the parent shell.
    JOB_ID=$(
        N_SIMS_PER_NODE=1 \
        RUN_0_COMP="$COMP" \
        OUTPUT_ROOT="$BENCH_SIM_ROOT" \
        NSTEPS="$SETUP_NSTEPS" \
        MAXWARN=2 \
        GPUS_PER_NODE=1 \
        CPUS_PER_SIM=8 \
        SAVE_FORCES=0 \
        sbatch \
            --partition="$SETUP_PARTITION" \
            --time=04:00:00 \
            --cpus-per-task=8 \
            --mem=16G \
            --gres=gpu:1 \
            --export=ALL \
            "$SCRIPT_DIR/../bash/sbatch_simulations.sh" \
        | awk '{print $NF}'
    )
    SETUP_JOB_IDS+=("$JOB_ID")
    printf '  [setup] %s: job %s submitted\n' "$COMP" "$JOB_ID"
done

# ── Phase 2: submit benchmark points ─────────────────────────────────────────
DEPENDENCY_ARG=""
if [[ ${#SETUP_JOB_IDS[@]} -gt 0 ]]; then
    DEP_STR=$(IFS=:; echo "${SETUP_JOB_IDS[*]}")
    DEPENDENCY_ARG="--dependency=afterok:${DEP_STR}"
fi

TPRS_STR=$(IFS=:; echo "${REFERENCE_TPRS[*]}")

printf '\n'
BATCH_COUNT=0
PREV_BENCH_JOB_ID=""    # for afterany chaining (minimises concurrent pending bench jobs)

while IFS=$'\t' read -r LABEL SIMS GPUS CPUS MEM POINT_PARTITION REST; do
    [[ -z "$LABEL" || "$LABEL" == \#* ]] && continue

    # CPU points keep their declared partition.  For GPU points the CLI
    # --partition is the default override (gpu_test ↔ gpu swap for dev), but
    # any row that explicitly declares a non-gpu_test partition (e.g. gpu for
    # full-node 8-GPU points) is honoured as-is — otherwise an 8-GPU row would
    # silently get redirected to gpu_test and tripped by the 4-GPU cap below.
    if [[ "$GPUS" == "0" ]]; then
        EFFECTIVE_PARTITION="$POINT_PARTITION"
    elif [[ "$POINT_PARTITION" != "gpu_test" ]]; then
        EFFECTIVE_PARTITION="$POINT_PARTITION"
    else
        EFFECTIVE_PARTITION="$PARTITION"
    fi

    # gpu_test cap enforcement (informational; actual SLURM would reject anyway)
    if [[ "$EFFECTIVE_PARTITION" == "gpu_test" && "$GPUS" -gt 4 ]]; then
        printf '  WARNING: point %s requests %s GPUs but gpu_test caps at 4 — skipping\n' \
            "$LABEL" "$GPUS" >&2
        continue
    fi

    TOTAL_CPUS=$(( CPUS * SIMS ))
    MEM_NUM="${MEM%G}"
    TOTAL_MEM="$(( MEM_NUM * SIMS ))G"

    POINT_DIR="$BENCH_ROOT/points/$LABEL"
    mkdir -p "$POINT_DIR"

    # Write metadata consumed by analyze_benchmark.py
    python3 -c "
import json, sys
meta = {
    'label': sys.argv[1],
    'sims_per_node': int(sys.argv[2]),
    'gpus_per_node': int(sys.argv[3]),
    'cpus_per_sim': int(sys.argv[4]),
    'mem_per_sim': sys.argv[5],
    'partition': sys.argv[6],
}
with open(sys.argv[7], 'w') as f:
    json.dump(meta, f, indent=2)
" "$LABEL" "$SIMS" "$GPUS" "$CPUS" "$MEM" "$EFFECTIVE_PARTITION" \
  "$POINT_DIR/point_meta.json"

    GRES_ARG=""
    [[ "${GPUS}" -gt 0 ]] && GRES_ARG="--gres=gpu:${GPUS}"

    BATCH_COUNT=$(( BATCH_COUNT + 1 ))

    if [[ "$DRY_RUN" -eq 1 ]]; then
        # In dry-run, build a printable command-line that includes the
        # variable assignments (helps the user eyeball what gets exported).
        printf '  [DRY RUN] BENCH_POINT_DIR=%s REFERENCE_TPRS=%s SIMS_PER_NODE=%s GPUS_PER_NODE=%s CPUS_PER_SIM=%s NSTEPS=%s ' \
            "$POINT_DIR" "$TPRS_STR" "$SIMS" "$GPUS" "$CPUS" "$NSTEPS"
        printf 'sbatch --partition=%s --time=%s --cpus-per-task=%s --mem=%s %s %s --export=ALL %s\n' \
            "$EFFECTIVE_PARTITION" "$TIME_LIMIT" "$TOTAL_CPUS" "$TOTAL_MEM" \
            "${GRES_ARG}" "${DEPENDENCY_ARG}" "$SCRIPT_DIR/sbatch_benchmark_hpc.sh"
        printf '    label=%s  sims=%s  gpus=%s  cpus/sim=%s  mem=%s  partition=%s\n' \
            "$LABEL" "$SIMS" "$GPUS" "$CPUS" "$TOTAL_MEM" "$EFFECTIVE_PARTITION"
    else
        # Compose the dependency string: afterok on Phase 1 setup jobs (so we
        # don't run benchmarks against a missing tpr) AND afterany on the
        # previous bench job (so SLURM only runs one bench point at a time —
        # keeps the pending count down under tight per-user QOS limits like
        # gpu_test's MaxSubmitJobs).
        POINT_DEP="$DEPENDENCY_ARG"
        if [[ -n "$PREV_BENCH_JOB_ID" ]]; then
            if [[ -n "$POINT_DEP" ]]; then
                POINT_DEP="${POINT_DEP},afterany:${PREV_BENCH_JOB_ID}"
            else
                POINT_DEP="--dependency=afterany:${PREV_BENCH_JOB_ID}"
            fi
        fi

        JOB_ID=$(
            BENCH_POINT_DIR="$POINT_DIR" \
            REFERENCE_TPRS="$TPRS_STR" \
            SIMS_PER_NODE="$SIMS" \
            GPUS_PER_NODE="$GPUS" \
            CPUS_PER_SIM="$CPUS" \
            NSTEPS="$NSTEPS" \
            sbatch \
                --partition="$EFFECTIVE_PARTITION" \
                --time="$TIME_LIMIT" \
                --cpus-per-task="$TOTAL_CPUS" \
                --mem="$TOTAL_MEM" \
                ${GRES_ARG:+"$GRES_ARG"} \
                ${POINT_DEP:+"$POINT_DEP"} \
                --export=ALL \
                "$SCRIPT_DIR/sbatch_benchmark_hpc.sh" \
            | awk '{print $NF}'
        )
        PREV_BENCH_JOB_ID="$JOB_ID"
        printf '  [bench]  %s: job %s  (sims=%s gpus=%s cpus/sim=%s)\n' \
            "$LABEL" "$JOB_ID" "$SIMS" "$GPUS" "$CPUS"
    fi

done < "$POINTS_FILE"

printf '\n%d benchmark job(s) queued\n' "$BATCH_COUNT"
printf 'After all jobs complete, run:\n'
printf '  python scripts/python/analyze_benchmark.py --root %s --recommend\n' "$BENCH_ROOT"
