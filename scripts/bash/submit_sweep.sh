#!/bin/bash
# submit_sweep.sh — freeze HP config at submission time and enqueue sbatch jobs.
#
# All hyperparameters are read from config.yaml NOW and baked into the sbatch
# call as FREEZE_* env vars. Queue wait cannot cause config drift — behavior
# is fully determined at submission time.
#
# Repeatable HP flags expand into a Cartesian product of jobs (one job per
# combination), mirroring run_sweep.py's SWEEP grid but running in parallel.
#
# Usage examples:
#
#   # Stage 5 confirmation — 3 parallel jobs, one seed group each:
#   bash scripts/bash/submit_sweep.sh --group stage_5_confirm \
#       --seeds "0 1" --seeds "2 3" --seeds "4"
#
#   # Stage 1b lr sweep — 3 lr values × 1 seed group = 3 parallel jobs:
#   bash scripts/bash/submit_sweep.sh --group stage_1b_tier_a_lr \
#       --lr "1e-5 1e-4 5e-4" \
#       --seeds "0 1"
#
#   # Stage 2b wd sweep — 3 wd values × 2 seed groups = 6 parallel jobs:
#   bash scripts/bash/submit_sweep.sh --group stage_2b_tier_a_wd \
#       --wd "1e-4 1e-3 1e-2" \
#       --seeds "0 1" --seeds "2 3"
#
#   # Override properties:
#   bash scripts/bash/submit_sweep.sh --group stage_0b_tier_a \
#       --properties "lipid_packing thickness thickness_std variation" \
#       --seeds "0 1 2 3 4"

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Parse arguments ───────────────────────────────────────────────────────────
GROUP=""
SEEDS_LIST=()
LR_LIST=()
WD_LIST=()
HIDDEN_DIM_LIST=()
NUM_LAYERS_LIST=()
PROPERTIES_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --group)        GROUP="$2";                                          shift 2 ;;
        --seeds)        SEEDS_LIST+=("$2");                                  shift 2 ;;
        --properties)   PROPERTIES_OVERRIDE="$2";                            shift 2 ;;
        # HP flags: space-separated values in one arg OR repeated flag both work.
        # e.g. --lr "1e-5 1e-4 5e-4"  OR  --lr 1e-5 --lr 1e-4 --lr 5e-4
        --lr)           read -ra _v <<< "$2"; LR_LIST+=("${_v[@]}");         shift 2 ;;
        --wd)           read -ra _v <<< "$2"; WD_LIST+=("${_v[@]}");         shift 2 ;;
        --hidden-dim)   read -ra _v <<< "$2"; HIDDEN_DIM_LIST+=("${_v[@]}"); shift 2 ;;
        --num-layers)   read -ra _v <<< "$2"; NUM_LAYERS_LIST+=("${_v[@]}"); shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GROUP" ]]; then
    echo "Usage: $0 --group <wandb_group> [--seeds \"0 1\"] [--lr 1e-4] [--wd 1e-3] ..." >&2
    exit 1
fi

# ── Defaults from config.yaml (frozen at submission time) ────────────────────
DEFAULT_HIDDEN_DIM=$(python scripts/python/print_config_var.py model.hidden_dim)
DEFAULT_NUM_LAYERS=$(python scripts/python/print_config_var.py model.num_layers)
DEFAULT_LR=$(python scripts/python/print_config_var.py training.learning_rate)
DEFAULT_WD=$(python scripts/python/print_config_var.py training.weight_decay)
DEFAULT_EPOCHS=$(python scripts/python/print_config_var.py training.epochs)
DEFAULT_PROPERTIES=$(python scripts/python/print_config_var.py vocab.active_properties)

# Fill lists with config defaults when no CLI override given
[[ ${#LR_LIST[@]}          -eq 0 ]] && LR_LIST=("$DEFAULT_LR")
[[ ${#WD_LIST[@]}          -eq 0 ]] && WD_LIST=("$DEFAULT_WD")
[[ ${#HIDDEN_DIM_LIST[@]}  -eq 0 ]] && HIDDEN_DIM_LIST=("$DEFAULT_HIDDEN_DIM")
[[ ${#NUM_LAYERS_LIST[@]}  -eq 0 ]] && NUM_LAYERS_LIST=("$DEFAULT_NUM_LAYERS")
[[ ${#SEEDS_LIST[@]}       -eq 0 ]] && SEEDS_LIST=("")   # empty → use run_sweep.py default

PROPERTIES="${PROPERTIES_OVERRIDE:-$DEFAULT_PROPERTIES}"

# ── Summary ───────────────────────────────────────────────────────────────────
N_JOBS=$(( ${#LR_LIST[@]} * ${#WD_LIST[@]} * ${#HIDDEN_DIM_LIST[@]} * ${#NUM_LAYERS_LIST[@]} * ${#SEEDS_LIST[@]} ))

echo "Submitting jobs for group: $GROUP  ($(date))"
echo "  hidden_dim : ${HIDDEN_DIM_LIST[*]}"
echo "  num_layers : ${NUM_LAYERS_LIST[*]}"
echo "  lr         : ${LR_LIST[*]}"
echo "  wd         : ${WD_LIST[*]}"
echo "  seeds      : ${SEEDS_LIST[*]:-"(run_sweep default)"}"
echo "  properties : ${PROPERTIES}"
echo "  epochs     : ${DEFAULT_EPOCHS}"
echo "  Total jobs : ${N_JOBS}"
echo ""

# ── Submit one job per HP combination ─────────────────────────────────────────
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

    JOB_ID=$(sbatch --export="$EXPORT_VARS" scripts/bash/sbatch_sweep.sh | awk '{print $NF}')
    echo "  Job ${JOB_ID}  h=${HIDDEN_DIM} l=${NUM_LAYERS} lr=${LR} wd=${WD} seeds=${SEEDS:-"(default)"}"

done
done
done
done
done

echo ""
echo "Monitor : squeue -u $USER"
echo "Logs    : logs/sweeps/sweep-<job_id>.out"
