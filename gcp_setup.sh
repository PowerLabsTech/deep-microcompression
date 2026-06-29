#!/usr/bin/env bash
# GCP VM setup — run once after the VM boots.
# Tested on: Deep Learning VM (DLVM) with CUDA 12.6, Python 3.12.
#
# Usage:
#   chmod +x gcp_setup.sh && ./gcp_setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_DIR/.venv"

echo "=========================================="
echo " Deep Microcompression — GCP Setup"
echo " Repo: $REPO_DIR"
echo "=========================================="

# ---------- Python venv -------------------------------------------------------
if [ ! -d "$VENV" ]; then
    echo "[1/4] Creating virtual environment …"
    python3 -m venv "$VENV"
else
    echo "[1/4] Virtual environment already exists — skipping."
fi

PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

# ---------- PyTorch (CUDA 12.6) -----------------------------------------------
echo "[2/4] Installing PyTorch 2.7 + CUDA 12.6 …"
"$PIP" install --upgrade pip --quiet
"$PIP" install \
    torch==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 \
    --quiet

# ---------- Other dependencies ------------------------------------------------
echo "[3/4] Installing remaining dependencies …"
"$PIP" install \
    tqdm \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    notebook \
    ipykernel \
    --quiet

"$PYTHON" -m ipykernel install --user --name deep-microcompression

# ---------- Verify GPU --------------------------------------------------------
echo "[4/4] Verifying GPU …"
"$PYTHON" - <<'EOF'
import torch
print(f"  PyTorch  : {torch.__version__}")
print(f"  CUDA     : {torch.version.cuda}")
print(f"  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND'}")
print(f"  Available: {torch.cuda.is_available()}")
EOF

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV/bin/activate"
echo ""
echo "Next: run ./gcp_run_nas.sh"
