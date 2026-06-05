#!/usr/bin/env bash
# Run NAS data generation for MobileNetV1/CIFAR-100 repeatedly.
#
# PREREQUISITE: train the baseline first by running reproduce.ipynb.
# The script will exit immediately if models/baseline.pth is missing.
#
# Usage:
#   ./run_nas.sh [NUM_RUNS] [NUM_DATA_PER_RUN] [WITH_TRAINING] [EPOCHS] [WIDTH_MULT]
#
# Defaults: 4 runs × 50 samples, with training, 2 epochs, α=0.5
# Total default: 200 samples.
#
# GPU strongly recommended — MobileNetV1 α=0.5 is ~3× faster than VGG-13.
#
# Examples:
#   ./run_nas.sh                      # 4 × 50, train, 2 epochs, α=0.5
#   ./run_nas.sh 4 50 false 0 0.5     # no training (fast, CPU feasible)
#   ./run_nas.sh 4 50 true  2 0.25    # smaller model (α=0.25)

set -euo pipefail
cd "$(dirname "$0")"

NUM_RUNS=${1:-4}
NUM_DATA=${2:-50}
WITH_TRAINING=${3:-true}
EPOCHS=${4:-2}
WIDTH_MULT=${5:-0.5}

PYTHON="python3"
VENV_PY="../../../.venv/bin/python"
[ -f "$VENV_PY" ] && PYTHON="$VENV_PY"

# Guard: baseline checkpoint must exist
BASELINE="models/baseline.pth"
if [ ! -f "$BASELINE" ]; then
    echo "ERROR: $BASELINE not found."
    echo "Please run reproduce.ipynb first to train and save the baseline model."
    exit 1
fi

echo "=========================================="
echo " MobileNetV1 (α=$WIDTH_MULT) / CIFAR-100 NAS"
echo " Runs:    $NUM_RUNS"
echo " Samples: $NUM_DATA per run  (total: $((NUM_RUNS * NUM_DATA)))"
echo " Train:   $WITH_TRAINING  |  Epochs: $EPOCHS"
echo " Python:  $PYTHON"
echo "=========================================="

for i in $(seq 1 "$NUM_RUNS"); do
    echo ""
    echo "--- Run $i / $NUM_RUNS ---"
    if [ "$WITH_TRAINING" = "true" ]; then
        "$PYTHON" generate_nas_data.py \
            --num_data   "$NUM_DATA"   \
            --epochs     "$EPOCHS"     \
            --width_mult "$WIDTH_MULT" \
            --with_training
    else
        "$PYTHON" generate_nas_data.py \
            --num_data   "$NUM_DATA"   \
            --epochs     "$EPOCHS"     \
            --width_mult "$WIDTH_MULT" \
            --no-with_training
    fi
done

echo ""
echo "=========================================="
echo " Done. NAS data files:"
ls -lh nas_data/*.pth 2>/dev/null || echo "  (none found)"
echo "=========================================="
