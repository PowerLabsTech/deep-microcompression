#!/usr/bin/env bash
# Run NAS data generation for MobileNetV2/CIFAR-100 repeatedly.
#
# Usage:
#   ./run_nas.sh [NUM_RUNS] [NUM_DATA_PER_RUN] [WITH_TRAINING] [EPOCHS]
#
# Defaults: 4 runs × 50 samples, with training, 2 epochs per sample.
# Total default: 200 samples.
#
# Note: GPU strongly recommended (large depthwise-separable convolutions).
# eval_subset=1024 keeps evaluation fast — each NAS step uses only 1024
# test images instead of the full 10,000.

set -euo pipefail
cd "$(dirname "$0")"

NUM_RUNS=${1:-4}
NUM_DATA=${2:-50}
WITH_TRAINING=${3:-true}
EPOCHS=${4:-2}
EVAL_SUBSET=${5:-1024}

PYTHON="python3"
VENV_PY="../../../.venv/bin/python"
[ -f "$VENV_PY" ] && PYTHON="$VENV_PY"

echo "=========================================="
echo " MobileNetV2 x0.5 / CIFAR-100 NAS"
echo " Runs:        $NUM_RUNS"
echo " Samples:     $NUM_DATA per run  (total: $((NUM_RUNS * NUM_DATA)))"
echo " Train:       $WITH_TRAINING  |  Epochs: $EPOCHS"
echo " Eval subset: $EVAL_SUBSET samples"
echo " Python:      $PYTHON"
echo "=========================================="

for i in $(seq 1 "$NUM_RUNS"); do
    echo ""
    echo "--- Run $i / $NUM_RUNS ---"
    if [ "$WITH_TRAINING" = "true" ]; then
        "$PYTHON" generate_nas_data.py \
            --num_data    "$NUM_DATA"   \
            --epochs      "$EPOCHS"     \
            --eval_subset "$EVAL_SUBSET" \
            --with_training
    else
        "$PYTHON" generate_nas_data.py \
            --num_data    "$NUM_DATA"   \
            --epochs      "$EPOCHS"     \
            --eval_subset "$EVAL_SUBSET" \
            --no-with_training
    fi
done

echo ""
echo "=========================================="
echo " Done. NAS data files:"
ls -lh nas_data/*.pth 2>/dev/null || echo "  (none found)"
echo "=========================================="
