#!/bin/bash
set -e

# Stop TensorFlow from printing info messages
export TF_CPP_MIN_LOG_LEVEL=2

echo "--- Setting up Python virtual environment (venv) ---"
python3 -m venv venv
source venv/bin/activate

echo "\n--- Installing dependencies (torch, torchvision, tqdm, tensorflow, numpy, matplotlib) ---"
pip install torch torchvision tqdm tensorflow numpy matplotlib

echo "\n--- Starting TFLite comparison (reproduce_table4.py) ---"
# Set a lucky number for reproducibility
python reproduce_table4.py

echo "\n--- Cleaning up ---"
deactivate

echo "\n--- TFLite comparison complete. ---"
echo "You can find the saved baseline Keras model in 'lenet5_model.keras'."
echo "Results are printed above."