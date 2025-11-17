import sys
import os
import copy
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import numpy as np
import time

# --- Add project root to path ---
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
    print("and the 'development' module is in the project root ('../../').")
    sys.exit(1)

# --- Constants ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

LUCKY_NUMBER = 10
BASELINE_MODEL_FILE = "lenet5_state_dict.pth"
INPUT_SHAPE = (1, 28, 28)

# --- Set random seed for reproducibility ---
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
    """Loads MNIST dataset."""
    print("Loading MNIST dataset...")
    data_transform = transforms.Compose([
        transforms.RandomCrop((24, 24)),
        transforms.Resize(INPUT_SHAPE[1:]),
        transforms.ToTensor(),
    ])
    
    # Use ../../../Datasets/ to match the path from the script's location
    mnist_train_dataset = datasets.MNIST("../../../Datasets/", train=True, download=True, transform=data_transform)
    mnist_test_dataset = datasets.MNIST("../../../Datasets/", train=False, download=True, transform=data_transform)
    
    num_workers = min(os.cpu_count(), 4) # Limit workers to avoid overhead # type: ignore
    
    mnist_train_loader = data.DataLoader(mnist_train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, drop_last=False, generator=torch.Generator().manual_seed(LUCKY_NUMBER))
    mnist_test_loader = data.DataLoader(mnist_test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, drop_last=False)
    
    return mnist_train_loader, mnist_test_loader

# --- 2. Define Model ---
def get_baseline_model():
    """Defines the LeNet-5 model structure used in the script."""
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
    """Calculates accuracy."""
    return (y_pred.argmax(dim=1) == y_true).sum().item() * 100

def train_baseline(model, train_loader, test_loader):
    """Trains or loads the baseline model."""
    print("\n--- STAGE 1: Training Baseline Model ---")
    if os.path.exists(BASELINE_MODEL_FILE):
        print(f"Loading existing baseline weights from {BASELINE_MODEL_FILE}...")
        model.load_state_dict(torch.load(BASELINE_MODEL_FILE, map_location=DEVICE), strict=False)
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
        metrics={"acc_prc": accuracy_fun}, # Use a different key to avoid conflicts
        callbacks=[early_stopper],
        device=DEVICE
    )
    
    print(f"Saving baseline weights to {BASELINE_MODEL_FILE}...")
    torch.save(model.cpu().state_dict(), BASELINE_MODEL_FILE)
    model.to(DEVICE)
    return model

def run_compression_experiment(baseline_model_state, compression_config, train_loader, test_loader, epochs=15):
    """
    Runs a full compression experiment (eval-pre, train, eval-post) for a given config.
    """
    config_str = str(compression_config)
    print(f"\n--- Running Config: {config_str} ---")
    
    is_static = compression_config.get("quantize", {}).get("scheme") == QuantizationScheme.STATIC
    calib_data = None
    
    # --- 1. Get Model Size and Pre-Training Accuracy ---
    model_pre = get_baseline_model()
    if is_static:
        model_pre = model_pre.fuse()
        
    model_pre.load_state_dict(baseline_model_state, strict=False)
    
    if is_static:
        # Get one batch of calibration data
        calib_data = next(iter(train_loader))[0].to(DEVICE)

    model_pre_eval = model_pre.init_compress(compression_config, INPUT_SHAPE, calib_data).to(DEVICE)
    
    pre_training_metric = model_pre_eval.evaluate(test_loader, {"acc": accuracy_fun}, device=DEVICE)
    pre_acc = pre_training_metric['acc']
    model_size = model_pre_eval.get_size_in_bits() // 8
    
    del model_pre, model_pre_eval # Free memory

    # --- 2. Get Post-Training (QAT / Retraining) Accuracy ---
    model_post = get_baseline_model()
    if is_static:
        model_post = model_post.fuse()
            
    model_post.load_state_dict(baseline_model_state, strict=False)

    # Initialize model *for training*
    model_post_train = model_post.init_compress(compression_config, INPUT_SHAPE, calib_data).to(DEVICE)

    print(f"Retraining/QAT ({epochs} epochs)...")
    criterion_fun = nn.CrossEntropyLoss()
    optimizer_fun = optim.SGD(model_post_train.parameters(), lr=1.e-3, weight_decay=5e-4, momentum=.9)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=2)

    model_post_train.fit(
        train_loader, epochs, 
        criterion_fun, optimizer_fun, lr_scheduler,
        validation_dataloader=test_loader, 
        metrics={"acc": accuracy_fun},
        device=DEVICE,
        verbose=False # Make it quiet for the loop
    )
    
    post_training_metric = model_post_train.evaluate(test_loader, {"acc": accuracy_fun}, device=DEVICE)
    post_acc = post_training_metric['acc']
    
    del model_post, model_post_train # Free memory
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    
    print(f"Result: Pre-Acc: {pre_acc:.2f}%, Post-Acc: {post_acc:.2f}%, Size: {model_size} bytes")
    
    return config_str, pre_acc, post_acc, model_size

# --- Main Execution ---
if __name__ == "__main__":
    
    start_time = time.time()
    
    # Get Data
    train_loader, test_loader = get_data_loaders()
    
    # --- STAGE 1: BASELINE ---
    baseline_model = get_baseline_model()
    baseline_model = train_baseline(baseline_model, train_loader, test_loader)
    
    print("Evaluating baseline model...")
    baseline_eval = baseline_model.evaluate(test_loader, {"acc": accuracy_fun}, device=DEVICE)
    original_size = baseline_model.get_size_in_bits() // 8
    baseline_acc = baseline_eval['acc']
    
    print(f"==> STAGE 1 (Baseline) COMPLETE ==")
    print(f"    Accuracy: {baseline_acc:.2f}%")
    print(f"    Size:     {original_size} bytes")
    
    # Store the baseline state dict to reload for each experiment
    baseline_state_dict = copy.deepcopy(baseline_model.state_dict())
    del baseline_model # Free memory
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    results = []
    
    # --- STAGE 2: PRUNING EXPERIMENTS ---
    print("\n\n--- STARTING PRUNING EXPERIMENTS ---")
    sparsities = np.arange(0.1, 1.0, 0.1)
    
    for sp in sparsities:
        compression_config = {
            "prune_channel": {
                "sparsity": sp,
                "metric": "l2"
            }
        }
        # Retrain for 20 epochs as per your script
        config_str, pre_acc, post_acc, model_size = run_compression_experiment(
            baseline_state_dict, compression_config, train_loader, test_loader, epochs=20
        )
        results.append((f"Pruning (Sparsity {sp:.1f})", pre_acc, post_acc, model_size))

    # --- STAGE 3: QUANTIZATION EXPERIMENTS ---
    print("\n\n--- STARTING QUANTIZATION EXPERIMENTS ---")
    bitwidths = [8, 4, 2]
    schemes = [QuantizationScheme.DYNAMIC, QuantizationScheme.STATIC]
    granularities = [QuantizationGranularity.PER_TENSOR, QuantizationGranularity.PER_CHANNEL]
    
    for s in schemes:
        for g in granularities:
            for b in bitwidths:
                compression_config = {
                    "quantize": {
                        "scheme": s,
                        "granularity": g,
                        "bitwidth": b
                    }
                }
                
                # QAT for 15 epochs as per paper table
                config_str, pre_acc, post_acc, model_size = run_compression_experiment(
                    baseline_state_dict, compression_config, train_loader, test_loader, epochs=15
                )
                results.append((f"{s.name.title()} Quant ({g.name.title()}, {b}-bit)", pre_acc, post_acc, model_size))

    # --- FINAL RESULTS ---
    print("\n\n--- REPRODUCTION FINISHED: TABLE 3 (LeNet-5 on MNIST) ---")
    print(f"Baseline Accuracy: {baseline_acc:.2f}% | Baseline Size: {original_size} bytes")
    print("=" * 100)
    print(f"{'Compression Method':<35} | {'Size (bytes)':<15} | {'Size (%)':<10} | {'Acc (Pre-Train) %':<20} | {'Acc (Post-Train) %':<20}")
    print("-" * 100)
    
    for name, pre_acc, post_acc, size in results:
        size_perc = (size / original_size) * 100
        print(f"{name:<35} | {size:<15} | {size_perc:<10.2f} | {pre_acc:<20.2f} | {post_acc:<20.2f}")
    
    print("=" * 100)
    print(f"Total experiment time: {(time.time() - start_time) / 60:.2f} minutes")