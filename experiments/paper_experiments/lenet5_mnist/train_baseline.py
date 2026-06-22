#!/usr/bin/env python3
"""
Train and save a reproducible LeNet-5 baseline on MNIST.

Usage:
    python train_baseline.py
    python train_baseline.py --seed 42 --epochs 30
    python train_baseline.py --force   # overwrite existing baseline.pth
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_SHARED    = os.path.abspath(os.path.join(_HERE, "../shared"))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SHARED)

from lenet5      import get_model
from mnist       import get_data_loaders, get_metric
from train_utils import train_baseline


MODELS_DIR  = os.path.join(_HERE, "models")
BASELINE_PT = os.path.join(MODELS_DIR, "baseline.pth")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int,  default=25)
    parser.add_argument("--epochs", type=int,  default=30)
    parser.add_argument("--force",  action="store_true",
                        help="Overwrite existing baseline.pth")
    args = parser.parse_args()

    if os.path.exists(BASELINE_PT) and not args.force:
        print(f"baseline.pth already exists. Use --force to retrain.")
        sys.exit(0)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed: {args.seed} | Device: {device} | Epochs: {args.epochs}")

    print("Loading MNIST …")
    train_loader, test_loader = get_data_loaders()
    metric_fn = get_metric()

    print("Building LeNet-5 …")
    model = get_model().to(device)

    print(f"Training baseline for {args.epochs} epochs …")
    train_baseline(model, train_loader, test_loader, metric_fn,
                   epochs=args.epochs, device=device)

    results = model.evaluate(test_loader, {"accuracy": metric_fn}, device)
    acc = results["accuracy"]
    print(f"\nBaseline accuracy: {acc:.2f}%")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": args.seed, "accuracy": acc},
               BASELINE_PT)
    print(f"Saved → {BASELINE_PT}")


if __name__ == "__main__":
    main()
