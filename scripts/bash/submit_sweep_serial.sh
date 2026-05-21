#!/bin/bash
# submit_sweep_serial.sh — serial sibling of submit_sweep.sh.
#
# Submits one sbatch job per HP cell. Each job is a single GPU process that
# runs run_sweep.py once and iterates seeds sequentially. Use this when:
#
#   * the target partition does NOT support --gres (some Goethe GPU nodes
#     allocate the whole node instead of single-GPU slices)
#   * --gres=gpu:N nodes are unavailable / queued and a wider node pool is
#     worth the loss of per-node packing
#   * you want the historical "one HP cell per sbatch, seeds in series"
#     behaviour for direct comparison with pre-2026-05 runs
#
# By default no --gres flag is added to the sbatch call (so jobs land on
# non-gres GPU nodes). Pass --gres "gpu:1" to opt in.
#
# A "run" is one (hidden_dim, num_layers, lr, wd, seed-group) combination.
# Repeatable --seeds builds seed *groups*: each group becomes one job and is
# expanded into multiple sequential seeds inside run_sweep.py via
# SWEEP_SEEDS. All other HP flags expand into a Cartesian product of jobs.
#
# Usage examples:
#
#   # Stage 5 confirmation — 5 seeds, all in one job, run sequentially:
#   bash scripts/bash/submit_sweep_serial.sh --group stage_5_confirm \
#       --seeds "0 1 2 3 4"
#
#   # Same 5 seeds, split across 3 jobs (run in parallel across nodes,
#   # seeds-in-job in series):
#   bash scripts/bash/submit_sweep_serial.sh --group stage_5_confirm \
#       --seeds "0 1" --seeds "2 3" --seeds "4"
#
#   # Stage 1b lr sweep — 3 lr × 1 seed group = 3 jobs:
#   bash scripts/bash/submit_sweep_serial.sh --group stage_1b_tier_a_lr \
#       --lr "1e-5 1e-4 5e-4" --seeds "0 1"
#
#   # Opt in to --gres on a gres-enabled partition:
#   bash scripts/bash/submit_sweep_serial.sh --group probe \
#       --partition gpu --gres "gpu:1" --seeds "0"

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Parse arguments ───────────────────────────────────────────────────────────
GROUP=""
SEEDS_LIST=()                # array of *groups* (each entry may be "0 1 2")
LR_LIST=()
WD_LIST=()
HIDDEN_DIM_LIST=()
NUM_LAYERS_LIST=()
PROPERTIES_OVERRIDE=""
PARTITION=""
TIME_LIMIT="24:00:00"
CPUS=8
MEM="64G"
GRES=""                      # empty → no --gres flag on sbatch call
CHUNKS_DIR_OVERRIDE=""       # empty → sbatch_sweep_serial.sh default WORK path

while [[ $# -gt 0 ]]; do
    case $1 in
        --group)         GROUP="$2";                                          shift 2 ;;
        --properties)    PROPERTIES_OVERRIDE="$2";                            shift 2 ;;
        # --seeds: each invocation appends ONE seed group (kept as a string).
        # Repeat the flag to submit multiple jobs that each iterate their
        # own seed set sequentially.
        --seeds)         SEEDS_LIST+=("$2");                                  shift 2 ;;
        # HP flags: repeatable or space-separated; expand into Cartesian product.
        --lr)            read -ra _v <<< "$2"; LR_LIST+=("${_v[@]}");         shift 2 ;;
        --wd)            read -ra _v <<< "$2"; WD_LIST+=("${_v[@]}");         shift 2 ;;
        --hidden-dim)    read -ra _v <<< "$2"; HIDDEN_DIM_LIST+=("${_v[@]}"); shift 2 ;;
        --num-layers)    read -ra _v <<< "$2"; NUM_LAYERS_LIST+=("${_v[@]}"); shift 2 ;;
        # SLURM resource flags.
        --partition)     PARTITION="$2";                                      shift 2 ;;
        --time)          TIME_LIMIT="$2";                                     shift 2 ;;
        --cpus)          CPUS="$2";                                           shift 2 ;;
        --mem)           MEM="$2";                                            shift 2 ;;
        --gres)          GRES="$2";                                           shift 2 ;;
        # Absolute path to a preprocessed-graphs dir on the cluster (the dir
        # that directly contains train/ val/ test/). Forwarded as CHUNKS_DIR
        # via --export and consumed by lipid_gnn.config.load_config.
        --chunks-dir)    CHUNKS_DIR_OVERRIDE="$2";                            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GROUP" ]]; then
    echo "Usage: $0 --group <wandb_group> [--seeds \"0 1\"] [--lr ...] \\" >&2
    echo "         [--partition gpu] [--time HH:MM:SS] [--cpus 8] [--mem 64G] \\" >&2
    echo "         [--gres gpu:1]" >&2
    exit 1
fi

# ── Defaults from config.yaml (frozen at submission time) ────────────────────
DEFAULT_HIDDEN_DIM=$(python scripts/python/print_config_var.py model.hidden_dim)
DEFAULT_NUM_LAYERS=$(python scripts/python/print_config_var.py model.num_layers)
DEFAULT_LR=$(python scripts/python/print_config_var.py training.learning_rate)
DEFAULT_WD=$(python scripts/python/print_config_var.py training.weight_decay)
DEFAULT_EPOCHS=$(python scripts/python/print_config_var.py training.epochs)
DEFAULT_PROPERTIES=$(python scripts/python/print_config_var.py vocab.active_properties)
DEFAULT_PARTITION=$(python scripts/python/print_config_var.py hpc.partition_train)

[[ ${#LR_LIST[@]}         -eq 0 ]] && LR_LIST=("$DEFAULT_LR")
[[ ${#WD_LIST[@]}         -eq 0 ]] && WD_LIST=("$DEFAULT_WD")
[[ ${#HIDDEN_DIM_LIST[@]} -eq 0 ]] && HIDDEN_DIM_LIST=("$DEFAULT_HIDDEN_DIM")
[[ ${#NUM_LAYERS_LIST[@]} -eq 0 ]] && NUM_LAYERS_LIST=("$DEFAULT_NUM_LAYERS")
# Empty seed list → "" sentinel = use run_sweep.py's hard-coded default.
[[ ${#SEEDS_LIST[@]}      -eq 0 ]] && SEEDS_LIST=("")

PROPERTIES="${PROPERTIES_OVERRIDE:-$DEFAULT_PROPERTIES}"
[[ -z "$PARTITION" ]] && PARTITION="$DEFAULT_PARTITION"

N_JOBS=$(( ${#LR_LIST[@]} * ${#WD_LIST[@]} * ${#HIDDEN_DIM_LIST[@]} * ${#NUM_LAYERS_LIST[@]} * ${#SEEDS_LIST[@]} ))

# ── Summary ───────────────────────────────────────────────────────────────────
echo "Submitting jobs for group: $GROUP  ($(date))"
echo "  mode       : serial (1 run per sbatch, seeds in series within job)"
echo "  partition  : $PARTITION"
echo "  time       : $TIME_LIMIT"
echo "  gres       : ${GRES:-"(none — non-gres GPU nodes)"}"
echo "  chunks_dir : ${CHUNKS_DIR_OVERRIDE:-"(sbatch default WORK path)"}"
echo "  cpus       : $CPUS"
echo "  mem        : $MEM"
echo "  hidden_dim : ${HIDDEN_DIM_LIST[*]}"
echo "  num_layers : ${NUM_LAYERS_LIST[*]}"
echo "  lr         : ${LR_LIST[*]}"
echo "  wd         : ${WD_LIST[*]}"
echo "  seed groups: ${#SEEDS_LIST[@]}  → ${SEEDS_LIST[*]:-"(run_sweep default)"}"
echo "  properties : ${PROPERTIES}"
echo "  epochs     : ${DEFAULT_EPOCHS}"
echo "  Total jobs : ${N_JOBS}"
echo ""

# ── Submit one sbatch job per HP × seed-group combination ────────────────────
for HIDDEN_DIM in "${HIDDEN_DIM_LIST[@]}"; do
for NUM_LAYERS in "${NUM_LAYERS_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for SEEDS in "${SEEDS_LIST[@]}"; do

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",WANDB_GROUP=${GROUP}"
    EXPORT_VARS+=",FREEZE_HIDDEN_DIM=${HIDDEN_DIM}"
    EXPORT_VARS+=",FREEZE_NUM_LAYERS=${NUM_LAYERS}"
    EXPORT_VARS+=",FREEZE_LR=${LR}"
    EXPORT_VARS+=",FREEZE_WD=${WD}"
    EXPORT_VARS+=",FREEZE_EPOCHS=${DEFAULT_EPOCHS}"
    EXPORT_VARS+=",FREEZE_PROPERTIES=${PROPERTIES}"
    [[ -n "$SEEDS" ]] && EXPORT_VARS+=",SWEEP_SEEDS=${SEEDS}"
    [[ -n "$CHUNKS_DIR_OVERRIDE" ]] && EXPORT_VARS+=",CHUNKS_DIR=${CHUNKS_DIR_OVERRIDE}"

    SBATCH_ARGS=(
        --partition="$PARTITION"
        --time="$TIME_LIMIT"
        --cpus-per-task="$CPUS"
        --mem="$MEM"
        --export="$EXPORT_VARS"
    )
    [[ -n "$GRES" ]] && SBATCH_ARGS+=(--gres="$GRES")

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" scripts/bash/sbatch_sweep_serial.sh | awk '{print $NF}')
    echo "  Job ${JOB_ID}  h=${HIDDEN_DIM} l=${NUM_LAYERS} lr=${LR} wd=${WD} seeds=${SEEDS:-"(default)"}"

done; done; done; done; done

echo ""
echo "Monitor : squeue -u $USER"
echo "Logs    : logs/sweeps/sweep-<job_id>.{out,err}"
