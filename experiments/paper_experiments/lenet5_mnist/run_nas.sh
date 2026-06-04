#!/usr/bin/env bash
# Run NAS data generation for LeNet-5/MNIST repeatedly to accumulate samples.
#
# Usage:
#   ./run_nas.sh [NUM_RUNS] [NUM_DATA_PER_RUN] [WITH_TRAINING] [EPOCHS]
#
# Defaults: 4 runs × 50 samples, with training, 3 epochs per sample.
# Total default: 200 samples.
#
# Examples:
#   ./run_nas.sh               # 4 × 50 with training
#   ./run_nas.sh 10 50 true 3  # 10 × 50 with training (500 samples)
#   ./run_nas.sh 4 50 false 0  # 4 × 50 without training (fast)

set -euo pipefail
cd "$(dirname "$0")"

NUM_RUNS=${1:-4}
NUM_DATA=${2:-50}
WITH_TRAINING=${3:-true}
EPOCHS=${4:-3}

# Prefer project venv if present
PYTHON="python3"
VENV_PY="../../../.venv/bin/python"
[ -f "$VENV_PY" ] && PYTHON="$VENV_PY"

echo "=========================================="
echo " LeNet-5 / MNIST NAS data generation"
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
            --num_data "$NUM_DATA" \
            --epochs   "$EPOCHS"  \
            --with_training
    else
        "$PYTHON" generate_nas_data.py \
            --num_data "$NUM_DATA" \
            --epochs   "$EPOCHS"  \
            --no-with_training
    fi
done

echo ""
echo "=========================================="
echo " Done. NAS data files:"
ls -lh nas_data/*.pth 2>/dev/null || echo "  (none found)"
echo "=========================================="
