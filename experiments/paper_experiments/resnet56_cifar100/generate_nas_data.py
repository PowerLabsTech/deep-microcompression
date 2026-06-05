#!/usr/bin/env python3
"""
Generate NAS compression data samples for ResNet-56 on CIFAR-100.
Target board: ATmega2560 (Arduino Mega) — 8 KB SRAM, 256 KB Flash.

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
from development.experiments.resnet   import get_model
from development.experiments.cifar100 import get_data_loaders, get_metric
from hardware_specs import ATMEGA2560, make_nas_filter

# -------------------------------------------------------------------------
NAS_DATA_DIR = os.path.join(_HERE, "nas_data")
INPUT_SHAPE  = (3, 32, 32)
HARDWARE     = ATMEGA2560


def main():
    parser = argparse.ArgumentParser(description="NAS data generation: ResNet-56/CIFAR-100")
    parser.add_argument("--num_data",      type=int,  default=50)
    parser.add_argument("--seed",          type=int,  default=-1,
                        help="Random seed; -1 uses current timestamp")
    parser.add_argument("--with_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs",        type=int,  default=2,
                        help="Fine-tune epochs per sampled config (only if --with_training)")
    args = parser.parse_args()

    seed   = int(time.time()) if args.seed == -1 else args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Board: {HARDWARE['name']} | Device: {device} | "
          f"Samples: {args.num_data} | Train: {args.with_training} | "
          f"Epochs: {args.epochs} | Seed: {seed}")

    torch.manual_seed(seed)

    print("Loading CIFAR-100 …")
    train_loader, test_loader = get_data_loaders()
    # Loads pretrained ResNet-56 from torch hub and maps weights into DMC Sequential
    print("Loading pretrained ResNet-56 …")
    baseline_model = get_model().to(device)
    metric         = get_metric()
    calibration_data = next(iter(train_loader))[0].to(device)

    nas_filter = make_nas_filter(HARDWARE, INPUT_SHAPE)

    print(f"Generating {args.num_data} NAS samples …")
    os.makedirs(NAS_DATA_DIR, exist_ok=True)
    fname    = f"nas_{args.num_data}_{seed}_train{args.with_training}.pth"
    out_path = os.path.join(NAS_DATA_DIR, fname)

    nas_data = {}
    n_done   = 0

    if os.path.exists(out_path):
        data = torch.load(out_path, weights_only=False)
        assert data["with_training"] == args.with_training, \
            "Existing data has different training setting."
        assert data["epochs"] == args.epochs, \
            "Existing data has different epoch setting."
        assert data["hardware"] == HARDWARE["name"], \
            "Existing data was generated for different hardware."
        nas_data = data["nas_parameters"]
        n_done   = len(next(iter(nas_data.values()))) if nas_data else 0
        print(f"Resuming: {n_done}/{args.num_data} samples already done.")

    remaining = args.num_data - n_done
    if remaining <= 0:
        print(f"All {n_done} samples already generated — nothing to do.")
    else:
        for i in range(remaining):
            print(f"  Sample {n_done + i + 1}/{args.num_data} …")
            nas_params = get_nas_compression_data(
                baseline_model,
                INPUT_SHAPE,
                test_loader,
                metric,
                calibration_data,
                filter=nas_filter,
                device=device,
                num_data=1,
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
            for k, v in nas_params.items():
                nas_data.setdefault(k, []).extend(v)

            torch.save({
                "nas_parameters": nas_data,
                "num_data":       args.num_data,
                "seed":           seed,
                "with_training":  args.with_training,
                "epochs":         args.epochs,
                "hardware":       HARDWARE["name"],
            }, out_path)

    n_saved = len(next(iter(nas_data.values()))) if nas_data else 0
    print(f"Saved {n_saved} samples → {out_path}")


if __name__ == "__main__":
    main()
