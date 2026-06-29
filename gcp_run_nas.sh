#!/usr/bin/env bash
# Master NAS data generation script — run on the GCP VM after gcp_setup.sh.
#
# Pipeline per model:
#   1. train_baselines.py   (only LeNet-5 and MobileNetV1 — others use hub weights)
#   2. generate_nas_data.py (repeated NUM_RUNS times to accumulate samples)
#
# Usage:
#   ./gcp_run_nas.sh                        # all models, default settings
#   ./gcp_run_nas.sh --num_runs 4           # 4 runs × 50 samples = 200 per model
#   ./gcp_run_nas.sh --models resnet56 vgg13   # skip the long-training models
#   ./gcp_run_nas.sh --no_training          # fast zero-shot (no fine-tune per sample)
#
# Outputs land in experiments/paper_experiments/<model>/nas_data/*.pth
# Logs are tee'd to logs/<model>_<timestamp>.log

set -euo pipefail
cd "$(dirname "$0")"

REPO_DIR="$(pwd)"
EXPERIMENTS="$REPO_DIR/experiments/paper_experiments"
VENV_PY="$REPO_DIR/.venv/bin/python"
PYTHON="${VENV_PY:-python3}"
[ ! -f "$VENV_PY" ] && PYTHON="python3"

LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# ---------- Defaults ----------------------------------------------------------
NUM_RUNS=4
NUM_DATA=50          # samples per run
EPOCHS=2             # fine-tune epochs per NAS sample (--with_training)
WITH_TRAINING=true
# All models in dependency order (pretrained first, trained-from-scratch last)
ALL_MODELS=(resnet56 vgg13 mobilenetv2 lenet5 mobilenetv1)

# ---------- Argument parsing --------------------------------------------------
SELECTED_MODELS=("${ALL_MODELS[@]}")

usage() {
    echo "Usage: $0 [--num_runs N] [--num_data N] [--epochs N] [--no_training] [--models m1 m2 ...]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_runs)   NUM_RUNS="$2";   shift 2 ;;
        --num_data)   NUM_DATA="$2";   shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --no_training) WITH_TRAINING=false; shift ;;
        --models)
            shift
            SELECTED_MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SELECTED_MODELS+=("$1"); shift
            done
            ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

TOTAL=$((NUM_RUNS * NUM_DATA))

echo "=========================================="
echo " Deep Microcompression — NAS Data Generation"
echo " Models : ${SELECTED_MODELS[*]}"
echo " Runs   : $NUM_RUNS × $NUM_DATA samples = $TOTAL per model"
echo " Train  : $WITH_TRAINING  |  Epochs: $EPOCHS"
echo " Python : $PYTHON"
echo "=========================================="

# ---------- Step 1: train baselines that need it ------------------------------
NEEDS_TRAINING=()
for m in "${SELECTED_MODELS[@]}"; do
    [[ "$m" == "lenet5" || "$m" == "mobilenetv1" ]] && NEEDS_TRAINING+=("$m")
done

if [ ${#NEEDS_TRAINING[@]} -gt 0 ]; then
    echo ""
    echo "--- Step 1: Training baselines: ${NEEDS_TRAINING[*]} ---"
    LOG="$LOG_DIR/train_baselines_${TS}.log"
    "$PYTHON" "$EXPERIMENTS/train_baselines.py" \
        --models "${NEEDS_TRAINING[@]}" \
        2>&1 | tee "$LOG"
    echo "Baseline log → $LOG"
else
    echo ""
    echo "--- Step 1: No baseline training needed (all models use hub weights) ---"
fi

# ---------- Step 2: run generate_nas_data.py for each model ------------------
declare -A MODEL_DIR=(
    [lenet5]="lenet5_mnist"
    [mobilenetv1]="mobilenetv1_cifar100"
    [mobilenetv2]="mobilenetv2_cifar100"
    [resnet56]="resnet56_cifar100"
    [vgg13]="vgg13_cifar100"
)

for MODEL in "${SELECTED_MODELS[@]}"; do
    DIR="$EXPERIMENTS/${MODEL_DIR[$MODEL]}"
    SCRIPT="$DIR/generate_nas_data.py"

    if [ ! -f "$SCRIPT" ]; then
        echo "WARNING: $SCRIPT not found — skipping $MODEL"; continue
    fi

    echo ""
    echo "=========================================="
    echo " Model: $MODEL  ($NUM_RUNS runs × $NUM_DATA samples)"
    echo "=========================================="
    LOG="$LOG_DIR/${MODEL}_nas_${TS}.log"

    for i in $(seq 1 "$NUM_RUNS"); do
        echo ""
        echo "--- $MODEL: run $i / $NUM_RUNS ---"
        if [ "$WITH_TRAINING" = "true" ]; then
            "$PYTHON" "$SCRIPT" \
                --num_data "$NUM_DATA" \
                --epochs   "$EPOCHS"  \
                --with_training 2>&1 | tee -a "$LOG"
        else
            "$PYTHON" "$SCRIPT" \
                --num_data "$NUM_DATA" \
                --epochs   "$EPOCHS"  \
                --no-with_training 2>&1 | tee -a "$LOG"
        fi
    done

    echo ""
    echo "$MODEL NAS data:"
    ls -lh "$DIR/nas_data/"*.pth 2>/dev/null || echo "  (none found)"
    echo "Log → $LOG"
done

echo ""
echo "=========================================="
echo " All done. NAS data summary:"
for MODEL in "${SELECTED_MODELS[@]}"; do
    DIR="$EXPERIMENTS/${MODEL_DIR[$MODEL]}/nas_data"
    COUNT=$(ls "$DIR"/*.pth 2>/dev/null | wc -l)
    echo "  $MODEL: $COUNT file(s) in $DIR"
done
echo "=========================================="
