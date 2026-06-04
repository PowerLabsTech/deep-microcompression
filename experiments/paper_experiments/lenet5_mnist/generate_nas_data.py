#!/usr/bin/env python3
"""
Generate NAS compression data samples for LeNet-5 on MNIST.
Target board: ATmega328P (Arduino Uno) — 2 KB SRAM, 32 KB Flash.

Usage:
    python generate_nas_data.py --num_data 50
    python generate_nas_data.py --num_data 50 --seed 42 --no-with_training
"""
import argparse
import os
import sys
import time

import torch
from torch import nn

# --- resolve project root ------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT  = os.path.abspath(os.path.join(_HERE, "../../.."))
_SHARED     = os.path.abspath(os.path.join(_HERE, "../shared"))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SHARED)

from development import get_nas_compression_data
from development.experiments.lenet5  import get_model
from development.experiments.mnist   import get_data_loaders, get_metric
from hardware_specs import ATMEGA328P, make_nas_filter

# -------------------------------------------------------------------------
NAS_DATA_DIR = os.path.join(_HERE, "nas_data")
INPUT_SHAPE  = (1, 28, 28)
HARDWARE     = ATMEGA328P


def main():
    parser = argparse.ArgumentParser(description="NAS data generation: LeNet-5/MNIST")
    parser.add_argument("--num_data",      type=int,  default=50)
    parser.add_argument("--seed",          type=int,  default=-1,
                        help="Random seed; -1 uses current timestamp")
    parser.add_argument("--with_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs",        type=int,  default=3,
                        help="Fine-tune epochs per sampled config (only if --with_training)")
    args = parser.parse_args()

    seed   = int(time.time()) if args.seed == -1 else args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Board: {HARDWARE['name']} | Device: {device} | "
          f"Samples: {args.num_data} | Train: {args.with_training} | "
          f"Epochs: {args.epochs} | Seed: {seed}")

    torch.manual_seed(seed)

    print("Loading MNIST …")
    train_loader, test_loader = get_data_loaders()
    baseline_model = get_model().to(device)
    metric         = get_metric()
    calibration_data = next(iter(train_loader))[0].to(device)

    # LeNet-5 has no pretrained weights — must be trained first.
    # Load from models/ if available, otherwise train from scratch.
    baseline_ckpt = os.path.join(_HERE, "models", "baseline.pth")
    if os.path.exists(baseline_ckpt):
        print(f"Loading baseline weights from {baseline_ckpt}")
        baseline_model.load_state_dict(
            torch.load(baseline_ckpt, weights_only=True)["model"]
        )
    else:
        print("No baseline checkpoint found — please run reproduce.ipynb first to train and save the baseline.")
        sys.exit(1)

    nas_filter = make_nas_filter(HARDWARE, INPUT_SHAPE)

    print(f"Generating {args.num_data} NAS samples …")
    nas_params = get_nas_compression_data(
        baseline_model,
        INPUT_SHAPE,
        test_loader,
        metric,
        calibration_data,
        filter=nas_filter,
        device=device,
        num_data=args.num_data,
        train=args.with_training,
        train_dataloader=train_loader,
        epochs=args.epochs,
        criterion_fun=nn.CrossEntropyLoss(),
        random_seed=seed,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"momentum": 0.9, "weight_decay": 5e-4},
        lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_kwargs={"mode": "min", "patience": 1},
    )

    os.makedirs(NAS_DATA_DIR, exist_ok=True)
    fname    = f"nas_{args.num_data}_{seed}_train{args.with_training}.pth"
    out_path = os.path.join(NAS_DATA_DIR, fname)
    torch.save({
        "nas_parameters": nas_params,
        "num_data":       args.num_data,
        "seed":           seed,
        "with_training":  args.with_training,
        "epochs":         args.epochs,
        "hardware":       HARDWARE["name"],
    }, out_path)

    n_saved = len(list(nas_params.values())[0])
    print(f"Saved {n_saved} samples → {out_path}")


if __name__ == "__main__":
    main()
