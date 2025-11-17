#!/bin/bash
set -e

echo "--- Setting up Python virtual environment (venv) ---"
python3 -m venv venv
source venv/bin/activate

echo "\n--- Installing dependencies (torch, torchvision, tqdm, matplotlib) ---"
pip install torch torchvision tqdm matplotlib

echo "\n--- Starting experiment reproduction (reproduce_table1.py) ---"
python reproduce_table1.py

echo "\n--- Cleaning up ---"
deactivate

echo "\n--- Experiment reproduction complete. ---"
echo "You can find the saved baseline model weights in 'lenet5_state_dict.pth'."
echo "Results are printed above."