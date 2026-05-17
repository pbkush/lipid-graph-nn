#!/bin/bash
# submit_simulations.sh — orchestrator for Martini 3 bilayer simulations on Goethe-HLR.
#
# Resolves the composition list (explicit --compositions, --missing-from-grid,
# --queue-file, or --from-csv), packs compositions into SLURM batches, and
# submits one sbatch job
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
#   # Resimulate the legacy 70-system corpus with the modern M3 ITPs:
#   bash scripts/bash/submit_simulations.sh --from-csv resources/done.csv \
#       --prod-ns 1000 --partition gpu --time 8:00:00
#   # (inverse of --completed-csv: the CSV's canonical_name column IS the work
#   #  list, not the skip list.  Useful for standardising all simulations to
#   #  the same ITP definitions.)
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
# LEGACY_DATA_DIR: where the pre-pipeline 70-system corpus lives.  By
# default read from config (`paths.data_dir`), but env-override is supported
# so tests (and any user who wants to disable the legacy-skip) can point at
# an empty/nonexistent path.
LEGACY_DATA_DIR="${LEGACY_DATA_DIR:-$(python scripts/python/print_config_var.py paths.data_dir)}"

# Partition-dependent defaults are filled after arg parsing (we need to know
# --partition first to pick between hpc_defaults and hpc_defaults_cpu).  See
# the "Partition dispatch" block below.

# ── Argument parsing ─────────────────────────────────────────────────────────
COMPOSITIONS=()
MISSING_GRID=""
QUEUE_FILE=""
FROM_CSV=""
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
PIN=""             # empty → fall through to hpc_defaults.pin (or hpc_defaults_cpu.pin)
MEM_PER_SIM=""
MPI_RANKS_PER_SIM=""   # CPU only; ignored on GPU partitions
MAX_QUEUE=0            # 0 = no cap
COMPLETED_CSV=""       # path to a scan_completed_systems.py output CSV

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
        --from-csv)             FROM_CSV="$2";                shift 2 ;;
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
        --pin)                  PIN="$2";                     shift 2 ;;
        --max-queue)            MAX_QUEUE="$2";               shift 2 ;;
        --completed-csv)        COMPLETED_CSV="$2";           shift 2 ;;
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

# gmx mdrun -pin {on,off,auto}: fall through to the config default for this
# partition.  Don't go through _default(): it does exit 1 on a missing key
# (which is fine for required fields but not here — hpc_defaults_cpu doesn't
# declare 'pin' yet).  Query print_config_var.py directly and default to "on".
if [[ -z "$PIN" ]]; then
    if PIN=$(python scripts/python/print_config_var.py "${DEFAULTS_KEY}.pin" 2>/dev/null); then
        :   # got it from config
    else
        PIN="on"
    fi
    [[ -z "$PIN" ]] && PIN="on"
fi
case "$PIN" in
    on|off|auto) ;;
    *) echo "ERROR: --pin must be one of on|off|auto, got '$PIN'" >&2; exit 1 ;;
esac

# ── Validate source exclusivity ───────────────────────────────────────────────
N_SOURCES=0
[[ "${#COMPOSITIONS[@]}" -gt 0 ]] && N_SOURCES=$(( N_SOURCES + 1 )) || true
[[ -n "$MISSING_GRID" ]]          && N_SOURCES=$(( N_SOURCES + 1 )) || true
[[ -n "$QUEUE_FILE" ]]            && N_SOURCES=$(( N_SOURCES + 1 )) || true
[[ -n "$FROM_CSV" ]]              && N_SOURCES=$(( N_SOURCES + 1 )) || true

if [[ "$N_SOURCES" -gt 1 ]]; then
    echo "ERROR: --compositions, --missing-from-grid, --queue-file, and --from-csv are mutually exclusive" >&2
    exit 1
fi
if [[ "$N_SOURCES" -eq 0 ]]; then
    echo "ERROR: one of --compositions, --missing-from-grid, --queue-file, or --from-csv is required" >&2
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
    # GROUP comes from (in order): env var, config.yaml hpc.group.
    # Falls through to fail-fast only if both are missing/empty.
    GROUP="${GROUP:-$(python scripts/python/print_config_var.py hpc.group 2>/dev/null || true)}"
    : "${GROUP:?set hpc.group in config.yaml or 'export GROUP=...' before running, or pass --output-root}"
    OUTPUT_ROOT="/work/${GROUP}/${USER}/${WORK_SUBPATH}/${HPC_OUTPUT_SUBPATH}"
fi

# ── Resolve composition list ──────────────────────────────────────────────────
if [[ -n "$MISSING_GRID" ]]; then
    # Pass BOTH the new pipeline output root AND the legacy data root.
    # missing_compositions dedupes by canonical name, so any composition
    # already present in either tree is dropped before the bash script ever
    # sees it.  Legacy systems use the run/prun.xtc-fallback path (they
    # predate manifest.json); new ones use the manifest's overall_status.
    # If LEGACY_DATA_DIR doesn't exist on this machine (e.g. HPC where the
    # legacy data lives elsewhere), missing_compositions returns [] for it
    # — no error.
    MISSING_ARGS=(
        --grid "$MISSING_GRID"
        --format lines
        --output-roots "$OUTPUT_ROOT" "$LEGACY_DATA_DIR"
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
elif [[ -n "$FROM_CSV" ]]; then
    # Inverse of --completed-csv: use the canonical_name column AS the work
    # list, not as the skip list.  Designed for resimulating the legacy
    # 70-system corpus with the modern M3 ITPs (done.csv from
    # scan_completed_systems.py is the natural input), but works for any
    # CSV with a canonical_name column.
    if [[ ! -f "$FROM_CSV" ]]; then
        echo "ERROR: --from-csv $FROM_CSV does not exist" >&2
        exit 1
    fi
    readarray -t COMPOSITIONS < <(python -c "
import csv, sys
with open(sys.argv[1]) as fh:
    reader = csv.DictReader(fh)
    if 'canonical_name' not in (reader.fieldnames or []):
        print('ERROR: --from-csv missing required column: canonical_name', file=sys.stderr)
        sys.exit(2)
    for row in reader:
        name = (row.get('canonical_name') or '').strip()
        if name:
            print(name)
" "$FROM_CSV") || { echo "ERROR: failed to read --from-csv $FROM_CSV" >&2; exit 1; }
fi

# ── Filter against --completed-csv (if given) ────────────────────────────────
# Lets the user pre-scan locally, upload a small CSV to HPC, and have the
# submitter skip already-simulated compositions without needing the actual
# trajectory data present on HPC.  The CSV format is produced by
# scripts/python/scan_completed_systems.py: first column is the canonical
# composition name; other columns are informational.  Canonicalisation in
# the scan step means non-canonical legacy dirs (POPC10_DIPC90) are still
# matched against canonical grid output (DIPC90_POPC10).
if [[ -n "$COMPLETED_CSV" ]]; then
    if [[ ! -f "$COMPLETED_CSV" ]]; then
        echo "ERROR: --completed-csv $COMPLETED_CSV does not exist" >&2
        exit 1
    fi
    _SKIPSET=$(mktemp)
    python -c "
import csv, sys
with open(sys.argv[1]) as fh:
    reader = csv.DictReader(fh)
    if 'canonical_name' not in (reader.fieldnames or []):
        print('ERROR: --completed-csv missing required column: canonical_name', file=sys.stderr)
        sys.exit(2)
    for row in reader:
        name = (row.get('canonical_name') or '').strip()
        if name:
            print(name)
" "$COMPLETED_CSV" > "$_SKIPSET" || { rm -f "$_SKIPSET"; exit 1; }

    N_BEFORE="${#COMPOSITIONS[@]}"
    _FILTERED=()
    while IFS= read -r comp; do
        if ! grep -qxF "$comp" "$_SKIPSET"; then
            _FILTERED+=("$comp")
        fi
    done < <(printf '%s\n' "${COMPOSITIONS[@]}")
    COMPOSITIONS=("${_FILTERED[@]}")
    N_AFTER="${#COMPOSITIONS[@]}"
    rm -f "$_SKIPSET"

    DROPPED=$(( N_BEFORE - N_AFTER ))
    if (( DROPPED > 0 )); then
        echo "INFO: --completed-csv $COMPLETED_CSV dropped $DROPPED of $N_BEFORE comp(s)" >&2
    fi
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

# ── Per-partition QOS guard rails ────────────────────────────────────────────
# Hard caps mirroring Goethe-HLR QOSMaxSubmitJobPerUser limits.  Hitting these
# at sbatch-time produces a cryptic error; better to fail fast here with a
# clear instruction to use --max-queue or re-run after some jobs complete.
case "$PARTITION" in
    gpu_test)  PARTITION_JOB_CAP=2  ;;
    general1)  PARTITION_JOB_CAP=40 ;;
    *)         PARTITION_JOB_CAP=0  ;;   # 0 = no cap (gpu, test, ...)
esac

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
fi

if (( PARTITION_JOB_CAP > 0 )) && (( N_BATCHES > PARTITION_JOB_CAP )); then
    MAX_COMPS=$(( PARTITION_JOB_CAP * SIMS_PER_NODE ))
    cat >&2 <<EOF
ERROR: $PARTITION allows at most $PARTITION_JOB_CAP jobs per user (QOSMaxSubmitJobPerUser);
       this submission needs $N_BATCHES batches ($N_TOTAL comps at $SIMS_PER_NODE sims/node).

To submit only the first $MAX_COMPS compositions in alphabetical order, add:
    --max-queue $MAX_COMPS

After those finish, re-run the same command — already-simulated systems are
auto-skipped via --missing-from-grid, so subsequent runs pick up where you
left off.  Alternatively, reduce --sims-per-node or switch to a partition
with a higher MaxSubmitJobs (e.g. --partition gpu).
EOF
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
SOURCE_DESC=""
[[ -n "$MISSING_GRID" ]] && SOURCE_DESC="missing-from-grid ($MISSING_GRID)"
[[ -n "$QUEUE_FILE"   ]] && SOURCE_DESC="queue-file ($QUEUE_FILE)"
[[ -n "$FROM_CSV"     ]] && SOURCE_DESC="from-csv ($FROM_CSV)"
[[ -z "$SOURCE_DESC"  ]] && SOURCE_DESC="explicit --compositions"

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
echo "  pin            : $PIN"
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

    # Build env var list, write to a sourceable .env file, pass its path as
    # a POSITIONAL ARG to the worker.  Avoids SLURM --export entirely —
    # four prior attempts at env propagation failed on Goethe-HLR's slurm-wlm:
    #   1. `--export="ALL,VAR=val,..."`        silently drops list entries
    #   2. `--export "ALL,VAR=val,..."`        parses the second token wrong
    #   3. `env VAR=val sbatch --export=ALL`   env-prefix lost between sbatch
    #                                          and the spawned job (somehow)
    #   4. `export VAR=val` in subshell, `sbatch --export=ALL`   same problem
    #                                          as 3 on this particular SLURM
    #
    # Positional args ALWAYS reach the script (it's literally just argv on
    # the worker's process), so writing the env to a file and passing the
    # path as $1 is bulletproof regardless of SLURM env-propagation quirks.
    ENV_PRELIST=(
        "OUTPUT_ROOT=$OUTPUT_ROOT"
        "N_SIMS_PER_NODE=$N_SIMS"
        "MAXWARN=$MAXWARN"
        "SAVE_FORCES=$SAVE_FORCES"
        "GPUS_PER_NODE=$GPUS_PER_NODE"
        "CPUS_PER_SIM=$CPUS_PER_SIM"
        "MPI_RANKS_PER_SIM=$MPI_RANKS_PER_SIM"
        "PIN=$PIN"
    )
    [[ -n "$PROD_NS"    ]] && ENV_PRELIST+=("PROD_NS=$PROD_NS")
    [[ -n "$NSTEPS"     ]] && ENV_PRELIST+=("NSTEPS=$NSTEPS")
    [[ -n "$NSTEPS_EQ"  ]] && ENV_PRELIST+=("NSTEPS_EQ=$NSTEPS_EQ")
    [[ -n "$NSTEPS_MIN" ]] && ENV_PRELIST+=("NSTEPS_MIN=$NSTEPS_MIN")
    [[ -n "$NTOMP"      ]] && ENV_PRELIST+=("NTOMP=$NTOMP")
    for (( i=0; i<N_SIMS; i++ )); do
        ENV_PRELIST+=("RUN_${i}_COMP=${COMPOSITIONS[$((BATCH_START + i))]}")
    done

    GRES_ARG=""
    [[ "$GPUS_PER_NODE" -gt 0 ]] && GRES_ARG="--gres=gpu:${N_SIMS}"

    # Write the env file under logs/simulations/ (shared filesystem on HPC,
    # visible from both login node and compute nodes).  Use printf %q to
    # safely quote any shell-special characters in values.
    mkdir -p logs/simulations
    if [[ "$DRY_RUN" -eq 1 ]]; then
        ENV_FILE_PATH="logs/simulations/submit_env.DRYRUN.sh"
    else
        ENV_FILE_PATH=$(mktemp logs/simulations/submit_env.XXXXXX.sh)
        {
            echo "# auto-generated by submit_simulations.sh at $(date -Iseconds)"
            echo "# sourced by ${SBATCH_WORKER} at job startup"
            for kv in "${ENV_PRELIST[@]}"; do
                name="${kv%%=*}"
                value="${kv#*=}"
                printf 'export %s=%q\n' "$name" "$value"
            done
        } > "$ENV_FILE_PATH"
    fi

    SBATCH_CMD=(
        sbatch
        --partition="$PARTITION"
        --time="$TIME_LIMIT"
        --cpus-per-task="$TOTAL_CPUS"
        --mem="$TOTAL_MEM"
        --export=ALL
    )
    [[ -n "$GRES_ARG" ]] && SBATCH_CMD+=("$GRES_ARG")
    SBATCH_CMD+=("$SBATCH_WORKER" "$ENV_FILE_PATH")

    if [[ "$DRY_RUN" -eq 1 ]]; then
        # Render the would-be env-file body inline so the user can eyeball.
        echo "  [DRY RUN]  ${SBATCH_CMD[*]}"
        echo "  [DRY RUN]  env file would contain:"
        for kv in "${ENV_PRELIST[@]}"; do
            name="${kv%%=*}"
            value="${kv#*=}"
            printf '             export %s=%q\n' "$name" "$value"
        done
    else
        JOB_ID=$("${SBATCH_CMD[@]}" | awk '{print $NF}')
        echo -n "  Job ${JOB_ID}  batch $((b+1))/${N_BATCHES}  N_SIMS=${N_SIMS}  cpus=${TOTAL_CPUS}  mem=${TOTAL_MEM}"
        [[ -n "$GRES_ARG" ]] && echo -n "  gpus=${N_SIMS}" || echo -n "  (cpu-only)"
        echo "  env=${ENV_FILE_PATH}"
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
