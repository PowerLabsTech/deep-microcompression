#!/bin/bash
set -e

echo "--- Setting up Python virtual environment (venv) ---"
python3 -m venv venv
source venv/bin/activate

echo "\n--- Installing dependencies (torch, torchvision, tqdm, matplotlib) ---"
pip install torch torchvision tqdm matplotlib

echo "\n--- Starting experiment reproduction (reproduce_lenet5.py) ---"
# Set a lucky number for reproducibility in case the script doesn't
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python reproduce_lenet5.py

echo "\n--- Cleaning up ---"
deactivate

echo "\n--- Experiment reproduction complete. ---"
echo "You can find the saved baseline model weights in 'lenet5_state_dict.pth'."
echo "Results are printed above."