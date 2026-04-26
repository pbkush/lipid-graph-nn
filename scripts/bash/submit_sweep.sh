#!/bin/bash
# submit_sweep.sh — freeze HP config at submission time and enqueue sbatch jobs.
#
# All hyperparameters are read from config.yaml NOW and baked into the sbatch
# call as env vars. The job ignores config.yaml at execution time, so queue
# wait does not introduce config drift.
#
# Usage:
#   # Single job, all seeds sequential:
#   bash scripts/bash/submit_sweep.sh --group stage_5_confirm
#
#   # Three parallel jobs, one seed subset each:
#   bash scripts/bash/submit_sweep.sh --group stage_5_confirm \
#       --seeds "0 1" --seeds "2 3" --seeds "4"
#
#   # Override properties (default: active_properties from config.yaml):
#   bash scripts/bash/submit_sweep.sh --group stage_1b_tier_a_lr \
#       --properties "lipid_packing thickness thickness_std variation"

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Parse arguments ───────────────────────────────────────────────────────────
GROUP=""
SEEDS_LIST=()
PROPERTIES_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --group)       GROUP="$2";              shift 2 ;;
        --seeds)       SEEDS_LIST+=("$2");      shift 2 ;;
        --properties)  PROPERTIES_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GROUP" ]]; then
    echo "Usage: $0 --group <wandb_group> [--seeds \"0 1\"] [--properties \"prop1 prop2\"]" >&2
    exit 1
fi

# Default: one job, all seeds from SWEEP (read at submission time from script)
if [[ ${#SEEDS_LIST[@]} -eq 0 ]]; then
    SEEDS_LIST=("")   # empty string → SWEEP_SEEDS not set → run_sweep.py uses its default
fi

# ── Freeze HP values from config.yaml at submission time ─────────────────────
HIDDEN_DIM=$(python scripts/python/print_config_var.py model.hidden_dim)
NUM_LAYERS=$(python scripts/python/print_config_var.py model.num_layers)
LR=$(python scripts/python/print_config_var.py training.learning_rate)
WD=$(python scripts/python/print_config_var.py training.weight_decay)
EPOCHS=$(python scripts/python/print_config_var.py training.epochs)
PROPERTIES="${PROPERTIES_OVERRIDE:-$(python scripts/python/print_config_var.py vocab.active_properties)}"

echo "Submitting jobs for group: $GROUP"
echo "  Frozen config at $(date):"
echo "    hidden_dim=${HIDDEN_DIM}  num_layers=${NUM_LAYERS}"
echo "    lr=${LR}  wd=${WD}  epochs=${EPOCHS}"
echo "    properties=${PROPERTIES}"
echo "  Jobs: ${#SEEDS_LIST[@]}"
echo ""

# ── Submit one job per seed group ─────────────────────────────────────────────
for SEEDS in "${SEEDS_LIST[@]}"; do
    EXPORT_VARS="ALL"
    EXPORT_VARS+=",WANDB_GROUP=${GROUP}"
    EXPORT_VARS+=",FREEZE_HIDDEN_DIM=${HIDDEN_DIM}"
    EXPORT_VARS+=",FREEZE_NUM_LAYERS=${NUM_LAYERS}"
    EXPORT_VARS+=",FREEZE_LR=${LR}"
    EXPORT_VARS+=",FREEZE_WD=${WD}"
    EXPORT_VARS+=",FREEZE_EPOCHS=${EPOCHS}"
    EXPORT_VARS+=",FREEZE_PROPERTIES=${PROPERTIES}"
    if [[ -n "$SEEDS" ]]; then
        EXPORT_VARS+=",SWEEP_SEEDS=${SEEDS}"
    fi

    JOB_ID=$(sbatch --export="$EXPORT_VARS" scripts/bash/sbatch_sweep.sh | awk '{print $NF}')
    SEED_LABEL="${SEEDS:-"(default)"}"
    echo "  Submitted job ${JOB_ID}  seeds=${SEED_LABEL}"
done

echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    logs/sweeps/sweep-<job_id>.out"
