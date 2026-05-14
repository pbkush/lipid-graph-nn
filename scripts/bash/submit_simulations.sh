#!/bin/bash
# submit_simulations.sh — orchestrator for Martini 3 bilayer simulations on Goethe-HLR.
#
# Resolves the composition list (explicit --compositions, --missing-from-grid, or
# --queue-file), packs compositions into SLURM batches, and submits one sbatch job
# per batch.  All simulation knobs are frozen at submission time (no config drift
# during queue wait) — mirroring the submit_sweep.sh / sbatch_sweep.sh pattern.
#
# Usage examples:
#
#   # Submit all missing DPPC corner systems for 100 ns (default partition=gpu):
#   export GROUP=cellmembrane
#   bash scripts/bash/submit_simulations.sh --missing-from-grid dppc_corner --prod-ns 100
#
#   # Single composition dry-run:
#   bash scripts/bash/submit_simulations.sh --compositions DIPC100 --prod-ns 10 --dry-run
#
#   # Read a pre-built queue file (from print_work_queue.py --format lines):
#   bash scripts/bash/submit_simulations.sh --queue-file /tmp/queue.txt --nsteps 10000000
#
#   # CPU partition (general1):
#   bash scripts/bash/submit_simulations.sh --compositions DPPC100 --prod-ns 100 \
#       --partition general1 --gpus-per-node 0
#
#   # gpu_test smoke run:
#   bash scripts/bash/submit_simulations.sh --compositions DIPC100 --nsteps 5000 \
#       --partition gpu_test --time 01:00:00 --dry-run

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Read partition-independent config (frozen at invocation time) ────────────
DEFAULT_PARTITION=$(python scripts/python/print_config_var.py hpc.partition_train)
DEFAULT_MAXWARN=$(python scripts/python/print_config_var.py martini_pipeline.gmx.maxwarn)
WORK_SUBPATH=$(python scripts/python/print_config_var.py hpc.work_subpath)
HPC_OUTPUT_SUBPATH=$(python scripts/python/print_config_var.py martini_pipeline.hpc_output_subpath)

# Partition-dependent defaults are filled after arg parsing (we need to know
# --partition first to pick between hpc_defaults and hpc_defaults_cpu).  See
# the "Partition dispatch" block below.

# ── Argument parsing ─────────────────────────────────────────────────────────
COMPOSITIONS=()
MISSING_GRID=""
QUEUE_FILE=""
LIPIDS=()
STEP=10

PROD_NS=""
NSTEPS=""
SAVE_FORCES=0
MAXWARN="$DEFAULT_MAXWARN"
NSTEPS_EQ=""
NSTEPS_MIN=""
NTOMP=""           # empty → auto-computed in sbatch script as SLURM_CPUS_PER_TASK/N_SIMS

OUTPUT_ROOT_OVERRIDE=""
PARTITION="$DEFAULT_PARTITION"
TIME_LIMIT="24:00:00"
# Empty-string sentinels: set by CLI flags, else filled from the partition's
# config block (hpc_defaults for GPU, hpc_defaults_cpu for general1).
GPUS_PER_NODE=""
SIMS_PER_NODE=""
CPUS_PER_SIM=""
MEM_PER_SIM=""
MPI_RANKS_PER_SIM=""   # CPU only; ignored on GPU partitions
MAX_QUEUE=0            # 0 = no cap

DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --compositions)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                COMPOSITIONS+=("$1"); shift
            done ;;
        --missing-from-grid)    MISSING_GRID="$2";           shift 2 ;;
        --queue-file)           QUEUE_FILE="$2";              shift 2 ;;
        --lipids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LIPIDS+=("$1"); shift
            done ;;
        --step)                 STEP="$2";                    shift 2 ;;
        --prod-ns)              PROD_NS="$2";                 shift 2 ;;
        --nsteps)               NSTEPS="$2";                  shift 2 ;;
        --save-forces)          SAVE_FORCES=1;                shift ;;
        --maxwarn)              MAXWARN="$2";                 shift 2 ;;
        --nsteps-eq)            NSTEPS_EQ="$2";               shift 2 ;;
        --nsteps-min)           NSTEPS_MIN="$2";              shift 2 ;;
        --ntomp)                NTOMP="$2";                   shift 2 ;;
        --output-root)          OUTPUT_ROOT_OVERRIDE="$2";    shift 2 ;;
        --partition)            PARTITION="$2";               shift 2 ;;
        --time)                 TIME_LIMIT="$2";              shift 2 ;;
        --gpus-per-node)        GPUS_PER_NODE="$2";           shift 2 ;;
        --sims-per-node)        SIMS_PER_NODE="$2";           shift 2 ;;
        --cpus-per-sim)         CPUS_PER_SIM="$2";            shift 2 ;;
        --mem-per-sim)          MEM_PER_SIM="$2";             shift 2 ;;
        --mpi-ranks-per-sim)    MPI_RANKS_PER_SIM="$2";       shift 2 ;;
        --max-queue)            MAX_QUEUE="$2";               shift 2 ;;
        --dry-run)              DRY_RUN=1;                    shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Partition dispatch ───────────────────────────────────────────────────────
# Map --partition to (config defaults key, sbatch worker script).  Unknown
# partitions fail fast (Decision 58); the user adds a case row to extend.
case "$PARTITION" in
    gpu|gpu_test|test)
        DEFAULTS_KEY="martini_pipeline.hpc_defaults"
        SBATCH_WORKER="scripts/bash/sbatch_simulations.sh"
        IS_CPU_PARTITION=0
        ;;
    general1)
        DEFAULTS_KEY="martini_pipeline.hpc_defaults_cpu"
        SBATCH_WORKER="scripts/bash/sbatch_simulations_general1.sh"
        IS_CPU_PARTITION=1
        ;;
    *)
        cat >&2 <<EOF
ERROR: unknown partition: $PARTITION
  Known partitions: gpu, gpu_test, test, general1
  Add a case row to submit_simulations.sh's partition-dispatch block to extend.
EOF
        exit 1
        ;;
esac

# Fail-fast if the partition's defaults block is missing from config.yaml
# (Decision 62).  print_config_var.py exits non-zero on a missing key.
_default() {
    local key="$1" default_when_optional="${2-}"
    local val
    if val=$(python scripts/python/print_config_var.py "$key" 2>/dev/null); then
        echo "$val"
    elif [[ -n "$default_when_optional" ]]; then
        echo "$default_when_optional"
    else
        echo "ERROR: required config key missing: $key" >&2
        echo "  For --partition $PARTITION, populate '${DEFAULTS_KEY}' in config.yaml." >&2
        echo "  (Run benchmark_hpc_general1.sh first to calibrate, or paste a stub.)" >&2
        exit 1
    fi
}

# Fill any unset partition-dependent defaults from the matching config block.
[[ -z "$SIMS_PER_NODE" ]] && SIMS_PER_NODE=$(_default "${DEFAULTS_KEY}.sims_per_node")
[[ -z "$CPUS_PER_SIM"  ]] && CPUS_PER_SIM=$(_default "${DEFAULTS_KEY}.cpus_per_sim")
[[ -z "$MEM_PER_SIM"   ]] && MEM_PER_SIM=$(_default "${DEFAULTS_KEY}.mem_per_sim")

if [[ "$IS_CPU_PARTITION" -eq 1 ]]; then
    GPUS_PER_NODE=0     # general1 has no GPUs
    [[ -z "$MPI_RANKS_PER_SIM" ]] && MPI_RANKS_PER_SIM=$(_default "${DEFAULTS_KEY}.mpi_ranks_per_sim")
else
    [[ -z "$GPUS_PER_NODE" ]] && GPUS_PER_NODE=$(_default "${DEFAULTS_KEY}.gpus_per_node")
    # MPI_RANKS_PER_SIM has no effect on the GPU worker; keep it as-is if set
    # (so the user can pass it without harm), default to 1 otherwise.
    [[ -z "$MPI_RANKS_PER_SIM" ]] && MPI_RANKS_PER_SIM=1
fi

# ── Validate source exclusivity ───────────────────────────────────────────────
N_SOURCES=0
[[ "${#COMPOSITIONS[@]}" -gt 0 ]] && N_SOURCES=$(( N_SOURCES + 1 )) || true
[[ -n "$MISSING_GRID" ]]          && N_SOURCES=$(( N_SOURCES + 1 )) || true
[[ -n "$QUEUE_FILE" ]]            && N_SOURCES=$(( N_SOURCES + 1 )) || true

if [[ "$N_SOURCES" -gt 1 ]]; then
    echo "ERROR: --compositions, --missing-from-grid, and --queue-file are mutually exclusive" >&2
    exit 1
fi
if [[ "$N_SOURCES" -eq 0 ]]; then
    echo "ERROR: one of --compositions, --missing-from-grid, or --queue-file is required" >&2
    echo "       Run with --help for usage examples." >&2
    exit 1
fi

# ── Validate production length ────────────────────────────────────────────────
if [[ -z "$PROD_NS" && -z "$NSTEPS" ]]; then
    echo "ERROR: one of --prod-ns <float> or --nsteps <int> is required" >&2
    exit 1
fi
if [[ -n "$PROD_NS" && -n "$NSTEPS" ]]; then
    echo "ERROR: --prod-ns and --nsteps are mutually exclusive" >&2
    exit 1
fi

# ── Resolve output root ───────────────────────────────────────────────────────
if [[ -n "$OUTPUT_ROOT_OVERRIDE" ]]; then
    OUTPUT_ROOT="$OUTPUT_ROOT_OVERRIDE"
else
    : "${GROUP:?set GROUP to your Goethe-HLR group (e.g. export GROUP=cellmembrane), or use --output-root}"
    OUTPUT_ROOT="/work/${GROUP}/${USER}/${WORK_SUBPATH}/${HPC_OUTPUT_SUBPATH}"
fi

# ── Resolve composition list ──────────────────────────────────────────────────
if [[ -n "$MISSING_GRID" ]]; then
    MISSING_ARGS=(
        --grid "$MISSING_GRID"
        --format lines
        --output-roots "$OUTPUT_ROOT"
    )
    [[ ${#LIPIDS[@]} -gt 0 ]] && MISSING_ARGS+=(--lipids "${LIPIDS[@]}")
    [[ "$STEP" -ne 10 ]]      && MISSING_ARGS+=(--step "$STEP")

    _PWQ_OUT=$(mktemp)
    _PWQ_ERR=$(mktemp)
    if ! python scripts/simulation/print_work_queue.py "${MISSING_ARGS[@]}" \
            >"$_PWQ_OUT" 2>"$_PWQ_ERR"; then
        echo "ERROR: failed to resolve --missing-from-grid $MISSING_GRID" >&2
        cat "$_PWQ_ERR" >&2
        rm -f "$_PWQ_OUT" "$_PWQ_ERR"
        exit 1
    fi
    readarray -t COMPOSITIONS < "$_PWQ_OUT"
    rm -f "$_PWQ_OUT" "$_PWQ_ERR"
elif [[ -n "$QUEUE_FILE" ]]; then
    if [[ ! -f "$QUEUE_FILE" ]]; then
        echo "ERROR: --queue-file $QUEUE_FILE does not exist" >&2
        exit 1
    fi
    readarray -t COMPOSITIONS < <(grep -v '^\s*#' "$QUEUE_FILE" | grep -v '^\s*$')
fi

if [[ "${#COMPOSITIONS[@]}" -eq 0 ]]; then
    echo "(no missing compositions — queue is empty)" >&2
    exit 0
fi

# ── Validate composition names ────────────────────────────────────────────────
python -c "
import sys, os
sys.path.insert(0, '.')
from lipid_gnn.martini_pipeline.composition import parse_name
errors = []
for c in sys.argv[1:]:
    try:
        parse_name(c)
    except ValueError as e:
        errors.append(f'  {c}: {e}')
if errors:
    print('ERROR: invalid composition name(s):\n' + '\n'.join(errors), file=sys.stderr)
    sys.exit(1)
" "${COMPOSITIONS[@]}"

# ── Apply max-queue cap ───────────────────────────────────────────────────────
if [[ "$MAX_QUEUE" -gt 0 && "${#COMPOSITIONS[@]}" -gt "$MAX_QUEUE" ]]; then
    OVERFLOW=$(( ${#COMPOSITIONS[@]} - MAX_QUEUE ))
    echo "INFO: --max-queue $MAX_QUEUE applied; dropping $OVERFLOW composition(s)" >&2
    COMPOSITIONS=("${COMPOSITIONS[@]:0:$MAX_QUEUE}")
fi

# ── Batch packing ─────────────────────────────────────────────────────────────
N_TOTAL="${#COMPOSITIONS[@]}"
N_BATCHES=$(( (N_TOTAL + SIMS_PER_NODE - 1) / SIMS_PER_NODE ))

# ── gpu_test guard rails (mirror submit_sweep.sh) ────────────────────────────
if [[ "$PARTITION" == "gpu_test" ]]; then
    if [[ "$TIME_LIMIT" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
        REQ_SEC=$((10#${BASH_REMATCH[1]}*3600 + 10#${BASH_REMATCH[2]}*60 + 10#${BASH_REMATCH[3]}))
        if (( REQ_SEC > 8*3600 )); then
            echo "WARNING: gpu_test max time is 08:00:00; capping --time from $TIME_LIMIT to 08:00:00" >&2
            TIME_LIMIT="08:00:00"
        fi
    else
        echo "WARNING: gpu_test max time is 08:00:00; could not parse --time=$TIME_LIMIT (leaving as-is)" >&2
    fi
    if (( N_BATCHES > 2 )); then
        echo "ERROR: gpu_test allows at most 2 jobs; this submission needs $N_BATCHES batches" >&2
        echo "       ($N_TOTAL compositions at $SIMS_PER_NODE/node). Reduce --sims-per-node or use --partition gpu." >&2
        exit 1
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
SOURCE_DESC="${MISSING_GRID:+"missing-from-grid ($MISSING_GRID)"}${QUEUE_FILE:+"queue-file ($QUEUE_FILE)"}${MISSING_GRID:+}${QUEUE_FILE:+}"
[[ "${#COMPOSITIONS[@]}" -gt 0 && -z "$MISSING_GRID" && -z "$QUEUE_FILE" ]] && SOURCE_DESC="explicit --compositions"

PROD_DESC=""
[[ -n "$PROD_NS" ]] && PROD_DESC="${PROD_NS} ns"
[[ -n "$NSTEPS"  ]] && PROD_DESC="${NSTEPS} steps"

DRY_TAG=""
[[ "$DRY_RUN" -eq 1 ]] && DRY_TAG="  === DRY RUN — no sbatch will be submitted ==="

echo ""
echo "Martini simulation submission  ($(date +%Y-%m-%d\ %H:%M))${DRY_TAG:+ — $DRY_TAG}"
echo "  source         : ${SOURCE_DESC:-explicit}"
echo "  output root    : $OUTPUT_ROOT"
echo "  partition      : $PARTITION"
echo "  time           : $TIME_LIMIT"
echo "  sims-per-node  : $SIMS_PER_NODE"
echo "  cpus-per-sim   : $CPUS_PER_SIM"
echo "  mem-per-sim    : $MEM_PER_SIM"
echo "  gpus-per-node  : $GPUS_PER_NODE"
echo "  prod length    : $PROD_DESC"
echo "  Total comps    : $N_TOTAL"
echo "  Batches        : $N_BATCHES (up to $SIMS_PER_NODE sims/node)"
echo ""

# ── Memory calculation (mirrors submit_sweep.sh resource scaling) ─────────────
MEM_NUM="${MEM_PER_SIM%%[A-Za-z]*}"
MEM_UNIT="${MEM_PER_SIM##*[0-9]}"

# ── Submit (or dry-run) one sbatch per batch ──────────────────────────────────
for (( b=0; b<N_BATCHES; b++ )); do
    BATCH_START=$(( b * SIMS_PER_NODE ))
    BATCH_END=$(( BATCH_START + SIMS_PER_NODE ))
    (( BATCH_END > N_TOTAL )) && BATCH_END=$N_TOTAL
    N_SIMS=$(( BATCH_END - BATCH_START ))

    # CPU partition allocates cores per (sim × rank × omp_thread); GPU
    # partition allocates per (sim × omp_thread) since 1 sim = 1 GPU = 1 rank.
    if [[ "$IS_CPU_PARTITION" -eq 1 ]]; then
        TOTAL_CPUS=$(( CPUS_PER_SIM * MPI_RANKS_PER_SIM * N_SIMS ))
    else
        TOTAL_CPUS=$(( CPUS_PER_SIM * N_SIMS ))
    fi
    TOTAL_MEM="$(( MEM_NUM * N_SIMS ))${MEM_UNIT}"

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",OUTPUT_ROOT=${OUTPUT_ROOT}"
    EXPORT_VARS+=",N_SIMS_PER_NODE=${N_SIMS}"
    EXPORT_VARS+=",MAXWARN=${MAXWARN}"
    EXPORT_VARS+=",SAVE_FORCES=${SAVE_FORCES}"
    EXPORT_VARS+=",GPUS_PER_NODE=${GPUS_PER_NODE}"
    EXPORT_VARS+=",CPUS_PER_SIM=${CPUS_PER_SIM}"
    EXPORT_VARS+=",MPI_RANKS_PER_SIM=${MPI_RANKS_PER_SIM}"
    [[ -n "$PROD_NS" ]]   && EXPORT_VARS+=",PROD_NS=${PROD_NS}"
    [[ -n "$NSTEPS" ]]    && EXPORT_VARS+=",NSTEPS=${NSTEPS}"
    [[ -n "$NSTEPS_EQ" ]] && EXPORT_VARS+=",NSTEPS_EQ=${NSTEPS_EQ}"
    [[ -n "$NSTEPS_MIN" ]] && EXPORT_VARS+=",NSTEPS_MIN=${NSTEPS_MIN}"
    [[ -n "$NTOMP" ]]     && EXPORT_VARS+=",NTOMP=${NTOMP}"
    [[ "$SAVE_FORCES" -eq 1 ]] && EXPORT_VARS+=",SAVE_FORCES=1"

    for (( i=0; i<N_SIMS; i++ )); do
        EXPORT_VARS+=",RUN_${i}_COMP=${COMPOSITIONS[$((BATCH_START + i))]}"
    done

    GRES_ARG=""
    [[ "$GPUS_PER_NODE" -gt 0 ]] && GRES_ARG="--gres=gpu:${N_SIMS}"

    SBATCH_CMD=(
        sbatch
        --partition="$PARTITION"
        --time="$TIME_LIMIT"
        --cpus-per-task="$TOTAL_CPUS"
        --mem="$TOTAL_MEM"
        --export="$EXPORT_VARS"
    )
    [[ -n "$GRES_ARG" ]] && SBATCH_CMD+=("$GRES_ARG")
    SBATCH_CMD+=("$SBATCH_WORKER")

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [DRY RUN]  ${SBATCH_CMD[*]}"
    else
        JOB_ID=$("${SBATCH_CMD[@]}" | awk '{print $NF}')
        echo -n "  Job ${JOB_ID}  batch $((b+1))/${N_BATCHES}  N_SIMS=${N_SIMS}  cpus=${TOTAL_CPUS}  mem=${TOTAL_MEM}"
        [[ -n "$GRES_ARG" ]] && echo -n "  gpus=${N_SIMS}" || echo -n "  (cpu-only)"
        echo ""
    fi

    for (( i=0; i<N_SIMS; i++ )); do
        echo "    [slot $i]  ${COMPOSITIONS[$((BATCH_START + i))]}"
    done
    echo ""
done

if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "Monitor  : squeue -u $USER"
    echo "Logs     : logs/simulations/submit-<jobid>.out   (per-batch orchestrator)"
    echo "           <output_root>/<comp>/sim-<jobid>-gpu<i>.{out,err}  (per-sim)"
fi
