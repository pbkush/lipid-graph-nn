#!/bin/bash
# submit_sweep.sh — freeze HP config at submission time and enqueue sbatch jobs.
#
# All hyperparameters are read from config.yaml NOW and baked into the sbatch
# call as RUN_<i>_* env vars, one slot per parallel training run on a node.
# Queue wait cannot cause config drift — behavior is fully determined at
# submission time.
#
# A "run" is one (hidden_dim, num_layers, lr, wd, seed) combination. Repeated
# / space-separated HP flags expand into a Cartesian product of runs. Runs
# are packed onto GPU nodes (default 8 GPUs per node, one run per GPU) and
# launched in parallel inside a single sbatch job. If the run count exceeds
# the per-node GPU count, multiple sbatch jobs are submitted.
#
# Usage examples:
#
#   # Stage 5 confirmation — 5 seeds → 5 parallel runs on one node:
#   bash scripts/bash/submit_sweep.sh --group stage_5_confirm --seeds "0 1 2 3 4"
#
#   # Stage 1b lr sweep — 3 lr × 2 seeds = 6 runs on one node:
#   bash scripts/bash/submit_sweep.sh --group stage_1b_tier_a_lr \
#       --lr "1e-5 1e-4 5e-4" --seeds "0 1"
#
#   # Stage 2b wd sweep — 3 wd × 4 seeds = 12 runs → 2 nodes (8 + 4):
#   bash scripts/bash/submit_sweep.sh --group stage_2b_tier_a_wd \
#       --wd "1e-4 1e-3 1e-2" --seeds "0 1 2 3"
#
#   # Quick test on gpu_test partition (max 8h, max 2 jobs):
#   bash scripts/bash/submit_sweep.sh --group probe \
#       --partition gpu_test --time 02:00:00 --seeds "0 1"
#
#   # Custom resource scaling (per-GPU):
#   bash scripts/bash/submit_sweep.sh --group big \
#       --seeds "0 1 2 3" --cpus-per-gpu 4 --mem-per-gpu 32G

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
PARTITION=""
TIME_LIMIT="24:00:00"
GPUS_PER_NODE=8
CPUS_PER_GPU=8
MEM_PER_GPU="64G"

while [[ $# -gt 0 ]]; do
    case $1 in
        --group)         GROUP="$2";                                          shift 2 ;;
        --properties)    PROPERTIES_OVERRIDE="$2";                            shift 2 ;;
        # Repeatable / space-separated value flags.
        # e.g. --lr "1e-5 1e-4 5e-4"  OR  --lr 1e-5 --lr 1e-4
        --seeds)         read -ra _v <<< "$2"; SEEDS_LIST+=("${_v[@]}");      shift 2 ;;
        --lr)            read -ra _v <<< "$2"; LR_LIST+=("${_v[@]}");         shift 2 ;;
        --wd)            read -ra _v <<< "$2"; WD_LIST+=("${_v[@]}");         shift 2 ;;
        --hidden-dim)    read -ra _v <<< "$2"; HIDDEN_DIM_LIST+=("${_v[@]}"); shift 2 ;;
        --num-layers)    read -ra _v <<< "$2"; NUM_LAYERS_LIST+=("${_v[@]}"); shift 2 ;;
        # SLURM resource flags.
        --partition)     PARTITION="$2";                                      shift 2 ;;
        --time)          TIME_LIMIT="$2";                                     shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";                                  shift 2 ;;
        --cpus-per-gpu)  CPUS_PER_GPU="$2";                                   shift 2 ;;
        --mem-per-gpu)   MEM_PER_GPU="$2";                                    shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$GROUP" ]]; then
    echo "Usage: $0 --group <wandb_group> [--seeds \"0 1\"] [--lr ...] \\" >&2
    echo "         [--partition gpu] [--time HH:MM:SS] [--gpus-per-node 8] \\" >&2
    echo "         [--cpus-per-gpu 8] [--mem-per-gpu 64G]" >&2
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

# ── Build flat run list (one entry per Cartesian-product combination) ─────────
COMBOS=()
for HIDDEN_DIM in "${HIDDEN_DIM_LIST[@]}"; do
for NUM_LAYERS in "${NUM_LAYERS_LIST[@]}"; do
for LR in "${LR_LIST[@]}"; do
for WD in "${WD_LIST[@]}"; do
for SEED in "${SEEDS_LIST[@]}"; do
    COMBOS+=("${HIDDEN_DIM}|${NUM_LAYERS}|${LR}|${WD}|${SEED}")
done; done; done; done; done

N_TOTAL=${#COMBOS[@]}
N_BATCHES=$(( (N_TOTAL + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

# ── gpu_test partition guards (max 8h, max 2 sbatch jobs) ────────────────────
if [[ "$PARTITION" == "gpu_test" ]]; then
    # Time cap at 08:00:00 — only enforced when format is HH:MM:SS or H:MM:SS.
    if [[ "$TIME_LIMIT" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
        REQ_SEC=$((10#${BASH_REMATCH[1]}*3600 + 10#${BASH_REMATCH[2]}*60 + 10#${BASH_REMATCH[3]}))
        if (( REQ_SEC > 8*3600 )); then
            echo "WARNING: gpu_test max time is 08:00:00; capping --time from $TIME_LIMIT to 08:00:00" >&2
            TIME_LIMIT="08:00:00"
        fi
    else
        echo "WARNING: gpu_test max time is 08:00:00; could not parse --time=$TIME_LIMIT, leaving as-is (SLURM will reject if over limit)" >&2
    fi
    if (( N_BATCHES > 2 )); then
        echo "ERROR: gpu_test allows at most 2 jobs; this submission needs $N_BATCHES batches" >&2
        echo "       ($N_TOTAL runs at $GPUS_PER_NODE/node). Reduce HP grid or use --partition gpu." >&2
        exit 1
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "Submitting jobs for group: $GROUP  ($(date))"
echo "  partition  : $PARTITION"
echo "  time       : $TIME_LIMIT"
echo "  hidden_dim : ${HIDDEN_DIM_LIST[*]}"
echo "  num_layers : ${NUM_LAYERS_LIST[*]}"
echo "  lr         : ${LR_LIST[*]}"
echo "  wd         : ${WD_LIST[*]}"
echo "  seeds      : ${SEEDS_LIST[*]:-"(run_sweep default)"}"
echo "  properties : ${PROPERTIES}"
echo "  epochs     : ${DEFAULT_EPOCHS}"
echo "  Total runs : ${N_TOTAL}"
echo "  Batches    : ${N_BATCHES} (up to ${GPUS_PER_NODE} parallel runs/node)"
echo ""

# ── Submit one sbatch job per batch of up-to-GPUS_PER_NODE runs ──────────────
# Parse "64G" → num=64, unit=G. Total mem scales with #runs in this batch.
MEM_NUM="${MEM_PER_GPU%[A-Za-z]*}"
MEM_UNIT="${MEM_PER_GPU##*[0-9]}"

for ((b=0; b<N_BATCHES; b++)); do
    BATCH_START=$(( b * GPUS_PER_NODE ))
    BATCH_END=$(( BATCH_START + GPUS_PER_NODE ))
    (( BATCH_END > N_TOTAL )) && BATCH_END=$N_TOTAL
    N_RUNS=$(( BATCH_END - BATCH_START ))

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",WANDB_GROUP=${GROUP}"
    EXPORT_VARS+=",N_RUNS_PER_NODE=${N_RUNS}"
    EXPORT_VARS+=",FREEZE_EPOCHS=${DEFAULT_EPOCHS}"
    EXPORT_VARS+=",FREEZE_PROPERTIES=${PROPERTIES}"

    for ((i=0; i<N_RUNS; i++)); do
        IFS='|' read -r H L LR WD SEED <<< "${COMBOS[$((BATCH_START + i))]}"
        EXPORT_VARS+=",RUN_${i}_HIDDEN_DIM=${H}"
        EXPORT_VARS+=",RUN_${i}_NUM_LAYERS=${L}"
        EXPORT_VARS+=",RUN_${i}_LR=${LR}"
        EXPORT_VARS+=",RUN_${i}_WD=${WD}"
        [[ -n "$SEED" ]] && EXPORT_VARS+=",RUN_${i}_SEED=${SEED}"
    done

    TOTAL_CPUS=$(( CPUS_PER_GPU * N_RUNS ))
    TOTAL_MEM="$(( MEM_NUM * N_RUNS ))${MEM_UNIT}"

    JOB_ID=$(sbatch \
        --partition="$PARTITION" \
        --time="$TIME_LIMIT" \
        --gres="gpu:$N_RUNS" \
        --cpus-per-task="$TOTAL_CPUS" \
        --mem="$TOTAL_MEM" \
        --export="$EXPORT_VARS" \
        scripts/bash/sbatch_sweep.sh | awk '{print $NF}')

    echo "  Job ${JOB_ID}  batch $((b+1))/${N_BATCHES}  N_RUNS=${N_RUNS}  cpus=${TOTAL_CPUS}  mem=${TOTAL_MEM}"
    for ((i=0; i<N_RUNS; i++)); do
        IFS='|' read -r H L LR WD SEED <<< "${COMBOS[$((BATCH_START + i))]}"
        echo "    [GPU ${i}]  h=${H} l=${L} lr=${LR} wd=${WD} seed=${SEED:-"(default)"}"
    done
done

echo ""
echo "Monitor : squeue -u $USER"
echo "Logs    : logs/sweeps/sweep-<job_id>.out                (orchestrator)"
echo "          logs/sweeps/sweep-<job_id>-gpu<i>.{out,err}    (per-run)"
