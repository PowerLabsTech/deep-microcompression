"""
Baseline and compressed model training helpers.
"""

import copy
import json
import os

import torch
from torch import nn, optim


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def train_baseline(
    model,
    train_loader,
    test_loader,
    metric_fn,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 5,
):
    """
    Train a full-precision baseline model with AdamW + early stopping.
    Returns the trained model (best val-loss checkpoint restored).
    """
    from development import EarlyStopper

    early_stopper = EarlyStopper(
        monitor_metric="validation_loss",
        delta=1e-7,
        mode="min",
        patience=patience,
        restore_best_state_dict=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    model.fit(
        train_loader, epochs, criterion, optimizer,
        validation_dataloader=test_loader,
        metrics={"acc": metric_fn},
        callbacks=[early_stopper],
        device=device,
    )
    return model


# ---------------------------------------------------------------------------
# Compressed model training  (two-step: prune → prune+quant)
# ---------------------------------------------------------------------------

def train_compressed(
    baseline_model,
    compression_config: dict,
    input_shape: tuple,
    train_loader,
    test_loader,
    metric_fn,
    calibration_data: torch.Tensor,
    epochs: int = 40,
    device: str = "cpu",
    two_step: bool = True,
):
    """
    Apply compression_config to baseline_model and fine-tune.

    Two-step schedule (default):
      - Step 1 (prune_epochs): prune only, SGD lr=1e-2
      - Step 2 (quant_epochs): prune + quantize, SGD lr=5e-3

    Single-step (two_step=False): apply full config, SGD lr=5e-3.
    """
    def _sgd_fit(model, n_epochs, lr):
        crit  = nn.CrossEntropyLoss()
        optim_fn = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(optim_fn, mode="min", patience=2, factor=0.5)
        model.fit(
            train_loader, n_epochs, crit, optim_fn, sched,
            validation_dataloader=test_loader,
            metrics={"acc": metric_fn},
            device=device,
        )
        return model

    if two_step:
        prune_epochs = max(1, epochs // 2)
        quant_epochs = epochs - prune_epochs

        # --- Step 1: prune only ---
        prune_config = copy.deepcopy(compression_config)
        del prune_config["quantize"]
        pruned = baseline_model.init_compress(
            prune_config, input_shape,
            calibration_data=calibration_data, device=device,
        )
        _sgd_fit(pruned, prune_epochs, lr=1e-2)

        # --- Step 2: full config on pruned model ---
        compressed = pruned.init_compress(
            compression_config, input_shape,
            calibration_data=calibration_data, device=device,
        )
        _sgd_fit(compressed, quant_epochs, lr=5e-3)
    else:
        compressed = baseline_model.init_compress(
            compression_config, input_shape,
            calibration_data=calibration_data, device=device,
        )
        _sgd_fit(compressed, epochs, lr=5e-3)

    return compressed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, metric_fn, input_shape: tuple, device: str = "cpu") -> dict:
    """
    Returns a dict with accuracy (%), weight size (bytes), and workspace (bytes).
    Uses the fused model for size/workspace measurements.
    """
    fused     = model.fuse(device=device)
    accuracy  = fused.evaluate(test_loader, {"acc": metric_fn}, device=device)["acc"]
    size_b    = fused.get_size_in_bytes()
    workspace = fused.get_workspace_size(input_shape)
    return {
        "accuracy":        round(accuracy, 4),
        "size_bytes":      size_b,
        "workspace_bytes": workspace,
        "size_kb":         round(size_b    / 1024, 2),
        "workspace_kb":    round(workspace / 1024, 2),
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, out_dir: str, filename: str = "results.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


def print_results_table(results: dict):
    """Pretty-print a comparison table from the results dict."""
    header = f"{'Model':<20} {'Acc (%)':>8} {'Size (KB)':>10} {'Workspace (KB)':>15}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>8.2f} {r['size_kb']:>10.2f} {r['workspace_kb']:>15.2f}")
