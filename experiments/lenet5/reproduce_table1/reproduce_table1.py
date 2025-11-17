import sys
import os
import copy
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# This assumes the script is in 'project_root/experiments/lenet5/'
sys.path.append("../../../")

try:
    from development import (
        Sequential,
        Conv2d,
        BatchNorm2d,
        ReLU,
        ReLU6,
        MaxPool2d,
        Flatten,
        Linear,
        EarlyStopper,
        QuantizationGranularity,
        QuantizationScheme
    )
except ImportError:
    print("Error: Could not import the 'development' module.")
    print("Please ensure this script is run from 'experiments/lenet5/'")
    print("and the 'development' module is in the project root ('../../../').")
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

LUCKY_NUMBER = 10
BASELINE_MODEL_FILE = "lenet5_state_dict.pth"
INPUT_SHAPE = (1, 28, 28)

# Set random seed for reproducibility
torch.manual_seed(LUCKY_NUMBER)
if DEVICE == "cuda":
    torch.cuda.manual_seed(LUCKY_NUMBER)
torch.manual_seed(LUCKY_NUMBER)
torch.cuda.manual_seed(LUCKY_NUMBER)
# cuDNN determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)


# --- 1. Load Data ---
def get_data_loaders():
    print("Loading MNIST dataset...")
    data_transform = transforms.Compose([
        transforms.RandomCrop((24, 24)),
        transforms.Resize(INPUT_SHAPE[1:]),
        transforms.ToTensor(),
    ])
    
    mnist_train_dataset = datasets.MNIST("../../../Datasets/", train=True, download=True, transform=data_transform)
    mnist_test_dataset = datasets.MNIST("../../../Datasets/", train=False, download=True, transform=data_transform)
    
    mnist_train_loader = data.DataLoader(mnist_train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), drop_last=False) # type: ignore
    mnist_test_loader = data.DataLoader(mnist_test_dataset, batch_size=32, shuffle=False, num_workers=os.cpu_count(), drop_last=False) # type: ignore
    
    return mnist_train_loader, mnist_test_loader

# --- 2. Define Model ---
def get_baseline_model():
    return Sequential(
        Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, pad=(2, 2, 2, 2), bias=True),
        BatchNorm2d(num_features=6),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0),
        Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
        BatchNorm2d(num_features=16),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0),
        Flatten(),
        Linear(in_features=16*5*5, out_features=84, bias=True),
        ReLU(),
        Linear(in_features=84, out_features=10, bias=True)
    ).to(DEVICE)

# --- 3. Training & Evaluation Functions ---
def accuracy_fun(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).sum().item() * 100

def train_baseline(model, train_loader, test_loader):
    print("\n--- STAGE 1: Training Baseline Model ---")
    if os.path.exists(BASELINE_MODEL_FILE):
        print(f"Loading existing baseline weights from {BASELINE_MODEL_FILE}...")
        model.load_state_dict(torch.load(BASELINE_MODEL_FILE, weights_only=True), strict=False)
        model.to(DEVICE)
        return model

    print(f"No baseline weights found. Training from scratch (up to 20 epochs)...")
    early_stopper = EarlyStopper(
        metric_name="validation_loss",
        min_valid_diff=1e-7,
        mode="min",
        patience=4,
        restore_best_state_dict=True,
    )
    criterion_fun = nn.CrossEntropyLoss()
    optimizer_fun = optim.Adam(model.parameters(), lr=1.e-3)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=2)

    model.fit(
        train_loader, 20, 
        criterion_fun, optimizer_fun, lr_scheduler,
        validation_dataloader=test_loader, 
        metrics={"acc": accuracy_fun},
        callbacks=[early_stopper],
        device=DEVICE
    )
    
    print(f"Saving baseline weights to {BASELINE_MODEL_FILE}...")
    torch.save(model.cpu().state_dict(), BASELINE_MODEL_FILE)
    model.to(DEVICE)
    return model

def train_pruned(baseline_model, train_loader, test_loader):
    print("\n--- STAGE 2: Applying Pruning & Retraining ---")
    
    # Pruning parameters from paper (Table 1 / Sec 4.1.1)
    pruning_config = {
        "prune_channel": {
            "sparsity": {
                "conv2d_0": 0,
                "conv2d_1": 9,
                "linear_0": 50
            },
            "metric": "l2"
        }
    }
    
    print(f"Applying pruning config: {pruning_config['prune_channel']['sparsity']}")
    
    # Re-initialize model architecture with pruning
    pruned_model = copy.deepcopy(baseline_model)
    pruned_model = pruned_model.init_compress(pruning_config, INPUT_SHAPE).to(DEVICE)
    
    # Retrain (fine-tune) the pruned model
    print("Retraining pruned model (20 epochs)...")
    criterion_fun = nn.CrossEntropyLoss()
    optimizer_fun = optim.SGD(pruned_model.parameters(), lr=1.e-3, weight_decay=5e-4, momentum=.9)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=1)

    pruned_model.fit(
        train_loader, 20, 
        criterion_fun, optimizer_fun, lr_scheduler,
        validation_dataloader=test_loader, 
        metrics={"acc": accuracy_fun},
        device=DEVICE
    )
    return pruned_model

def train_quantized_pruned(pruned_model, train_loader, test_loader):
    print("\n--- STAGE 3: Applying Quantization (QAT) & Retraining ---")
    
    # Configs from paper (Table 2, 4-bit static) and notebook
    pruning_config = {
        "prune_channel": {
            "sparsity": {
                "conv2d_0": 0,
                "conv2d_1": 9,
                "linear_0": 50
            },
            "metric": "l2"
        }
    }
    quantization_config = {
        "quantize": {
            "scheme": QuantizationScheme.STATIC,
            "granularity": QuantizationGranularity.PER_TENSOR,
            "bitwidth": 4
        }
    }
    full_compression_config = {**pruning_config, **quantization_config}

    print(f"Applying quantization config: 4-bit, STATIC, PER_TENSOR")
    
    # Get one batch of calibration data
    calibration_data = next(iter(test_loader))[0].to(DEVICE)
    
    # Re-initialize model with *both* configs
    quantized_model = copy.deepcopy(pruned_model)
    quantized_model = quantized_model.to(DEVICE) # Fuse before QAT

    # Initialize compression for QAT
    quantized_model = quantized_model.init_compress(
        full_compression_config, 
        INPUT_SHAPE, 
        calibration_data
    ).to(DEVICE)

    # Perform Quantization-Aware Training (15 epochs from paper Table 2)
    print("Performing QAT (15 epochs)...")
    criterion_fun = nn.CrossEntropyLoss()
    optimizer_fun = optim.SGD(quantized_model.parameters(), lr=1.e-4, weight_decay=5e-4, momentum=.9)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=1)

    quantized_model.fit(
        train_loader, 20, 
        criterion_fun, optimizer_fun, lr_scheduler,
        validation_dataloader=test_loader, 
        metrics={"acc": accuracy_fun},
        device=DEVICE
    )
    return quantized_model

# --- Main Execution ---
if __name__ == "__main__":
    
    # Get Data
    train_loader, test_loader = get_data_loaders()
    
    # --- STAGE 1: BASELINE ---
    baseline_model = get_baseline_model()
    baseline_model = train_baseline(baseline_model, train_loader, test_loader)
    
    print("Evaluating baseline model...")
    baseline_eval = baseline_model.evaluate(test_loader, {"acc": accuracy_fun}, device=DEVICE)
    original_size = baseline_model.get_size_in_bits() // 8
    print(f"==> STAGE 1 (Baseline) COMPLETE ==")
    print(f"    Accuracy: {baseline_eval['acc']:.2f}%")
    print(f"    Size:     {original_size} bytes")

    # --- STAGE 2: PRUNED ---
    # Use a copy to keep the original baseline model clean
    pruned_model = train_pruned(baseline_model, train_loader, test_loader)
    
    print("Evaluating pruned model...")
    pruned_eval = pruned_model.evaluate(test_loader, {"acc": accuracy_fun}, device=DEVICE)
    pruned_size = pruned_model.get_size_in_bits() // 8
    print(f"\n==> STAGE 2 (Pruned) COMPLETE ==")
    print(f"    Accuracy: {pruned_eval['acc']:.2f}%")
    print(f"    Size:     {pruned_size} bytes ({pruned_size/original_size*100:.2f}% of original)")

    print(f"\n===> Layerwise Prunning Results:")
    for i, (name, module) in enumerate(pruned_model.names_layers()):
        if ("conv2d" in name) or ("linear" in name):
            print(f"    Layer name : {name}, Original size {baseline_model[i].get_size_in_bits()/8*1024} Reduced size {module.get_size_in_bits()/8*1024}:  Size Ratio: {(1 - module.get_size_in_bits()/baseline_model[i].get_size_in_bits())*100:.2f}%") # type: ignore


    # --- STAGE 3: QUANTIZED-PRUNED ---
    quantized_model = train_quantized_pruned(pruned_model, train_loader, test_loader)
    
    print("Evaluating final quantized-pruned model...")
    quantized_eval = quantized_model.evaluate(test_loader, {"acc": accuracy_fun}, device=DEVICE)
    quantized_size = quantized_model.get_size_in_bits() // 8
    print(f"\n==> STAGE 3 (Quantized-Pruned) COMPLETE ==")
    print(f"    Accuracy: {quantized_eval['acc']:.2f}%")
    print(f"    Size:     {quantized_size} bytes ({quantized_size/original_size*100:.2f}% of original)")

    print("\n--- REPRODUCTION FINISHED ---")
    print("\nFinal Results Summary:")
    print(f"Baseline:   {baseline_eval['acc']:.2f}% Acc, {original_size} bytes")
    print(f"Pruned:     {pruned_eval['acc']:.2f}% Acc, {pruned_size} bytes, {pruned_size/original_size*100:.2f}% of original")
    print(f"Quantized:  {quantized_eval['acc']:.2f}% Acc, {quantized_size} bytes, {quantized_size/original_size*100:.2f}% of original")