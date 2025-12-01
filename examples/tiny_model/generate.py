import torch
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("../../")

# DMC Imports
import development
from development import Sequential
from development import Conv2d, Linear, ReLU, Flatten
from development import QuantizationScheme, QuantizationGranularity

# Defining constants
MODEL_VAR_NAME = "tiny_model"
DEPLOYMENT_BASE_DIR = "deploy"

assert os.path.exists(DEPLOYMENT_BASE_DIR) and os.path.isdir(DEPLOYMENT_BASE_DIR), f"The project dir {DEPLOYMENT_BASE_DIR} does not exist, please create."

# 1. Setup Dummy Data (Simulating MNIST)
print("[1] Generating Dummy Data...")
input_shape = (1, 28, 28) # N, C, H, W
X_train = torch.randn(100, 1, 28, 28) # 100 images
y_train = torch.randint(0, 10, (100,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=10)

# Calibration data is crucial for Static Quantization to find Min/Max ranges
calibration_data = X_train[0:10] 

# 2. Define the Architecture
# Using DMC layers (Conv2d, Linear) instead of torch.nn
print("[2] Defining Model...")
model = Sequential(
    # Input: 1 channel, Output: 4 filters, 3x3 kernel
    Conv2d(1, 4, kernel_size=3, pad=(1, 1, 1, 1)), 
    ReLU(),
    Flatten(),
    Linear(4 * 28 * 28, 10) 
)
model_size = model.get_size_in_bytes()

# 3. Define Compression Configuration
# We apply Structured Pruning (Stage 1) and 4-bit Static Quantization (Stage 2)
config = {
    "prune_channel": {
        "sparsity": 0.5,  # Remove 50% of filters/neurons
        "metric": "l2"    # Rank importance by L2 norm
    },
    "quantize": {
        "scheme": QuantizationScheme.STATIC, # Pre-compute scales for integer-only inference
        "bitwidth": 4,                       # Aggressive 4-bit compression
        "granularity": QuantizationGranularity.PER_TENSOR
    }
}

# 4. Initialize Compression Pipeline
# This sets up the pruning masks and quantization observers
print("[3] Initializing Compression Pipeline...")
compressed_model = model.init_compress(
    config, 
    input_shape=input_shape, 
    calibration_data=calibration_data
)
compressed_model_size = compressed_model.get_size_in_bytes()

# 5. Fine-Tune (Retraining)
# Recover accuracy lost due to pruning/quantization noise (QAT)
print("[4] Fine-tuning (Retraining)...")
optimizer = torch.optim.SGD(compressed_model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

compressed_model.fit(
    train_dataloader=train_loader,
    epochs=2, # Short run for example
    criterion_fun=criterion,
    optimizer_fun=optimizer
)

# 6. Export to C (Stage 3)
# Generates the bit-packed C arrays and header files
print("[5] Exporting to C Code...")

src_dir = os.path.join(DEPLOYMENT_BASE_DIR, "src")
include_dir = os.path.join(DEPLOYMENT_BASE_DIR, "include")
os.makedirs(src_dir, exist_ok=True)
os.makedirs(include_dir, exist_ok=True)

compressed_model.eval()
test_input = X_train[:1]
compressed_model.convert_to_c(
    input_shape=input_shape,
    var_name=MODEL_VAR_NAME,
    src_dir=src_dir,
    include_dir=include_dir,
    test_input=test_input # Export one image for C++ verification
)

print(f"The expected output for for the test input is {compressed_model.output_quantize.apply(compressed_model(test_input)).tolist()}")
print(f"Model compressed from an initial size of {model_size/1024}KB to {compressed_model_size/1024}KB, a {(1 - (compressed_model_size/model_size))*100:.2f} reduction in size.")
print(f"Success! C files generated in {DEPLOYMENT_BASE_DIR}")