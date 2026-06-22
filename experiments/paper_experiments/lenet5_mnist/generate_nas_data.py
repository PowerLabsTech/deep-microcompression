#!/usr/bin/env python3
"""
Generate NAS compression data for LeNet-5/MNIST from a pre-generated config pool.
Target board: ATmega328P (Arduino Uno) — 2 KB SRAM, 32 KB Flash.

Each cloud job processes a deterministic slice of the shared config pool,
guaranteeing no duplicate configs across parallel workers.

Usage:
    # First generate the pool locally:
    python generate_config_pool.py --n_configs 1000

    # Then each cloud job runs with its assigned slice:
    python generate_nas_data.py --job_id 0  --pool_file config_pool.pth
    python generate_nas_data.py --job_id 1  --pool_file config_pool.pth
    ...
    python generate_nas_data.py --job_id 19 --pool_file config_pool.pth

Fine-tuning hyperparameters (empirically validated via nas_hp_benchmark.py):
    lr=1e-2, CosineAnnealingLR, 5 epochs  →  0.53pp mean error vs 20-epoch reference.
"""
import argparse
import os
import subprocess
import sys

import torch
from torch import nn


def _gcs_cp(src: str, dst: str) -> None:
    result = subprocess.run(["gsutil", "cp", src, dst], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gsutil cp failed\n  src: {src}\n  dst: {dst}\n  stderr: {result.stderr.strip()}"
        )

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
_SHARED    = os.path.abspath(os.path.join(_HERE, "../shared"))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _SHARED)

from development import get_nas_compression_data
from lenet5         import get_model
from mnist          import get_data_loaders, get_metric
from hardware_specs import ATMEGA328P

NAS_DATA_DIR = os.path.join(_HERE, "nas_data")
INPUT_SHAPE  = (1, 28, 28)
HARDWARE     = ATMEGA328P

# Empirically validated fine-tuning hyperparameters (see nas_hp_benchmark.py)
LR           = 1e-2
EPOCHS       = 5
ETA_MIN      = 1e-5
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4


def main():
    parser = argparse.ArgumentParser(description="NAS data generation: LeNet-5/MNIST (pool-based)")
    parser.add_argument("--start",          type=int, required=True,
                        help="First config index to process (inclusive)")
    parser.add_argument("--end",            type=int, required=True,
                        help="Last config index to process (exclusive)")
    parser.add_argument("--pool_file",      type=str, default="gcp_nas_config_pool.pth",
                        help="Local pool file name (or full path)")
    parser.add_argument("--pool_gcs_uri",   type=str, default=None,
                        help="GCS URI to download the pool file from (cloud runs)")
    parser.add_argument("--output_gcs_dir", type=str, default=None,
                        help="GCS URI prefix to upload results to after completion")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download pool from GCS if running in the cloud
    if args.pool_gcs_uri:
        local_pool = f"/tmp/{os.path.basename(args.pool_gcs_uri)}"
        print(f"Downloading pool from {args.pool_gcs_uri} …")
        _gcs_cp(args.pool_gcs_uri, local_pool)
        pool_path = local_pool
    else:
        pool_path = os.path.join(_HERE, args.pool_file)

    if not os.path.exists(pool_path):
        print(f"ERROR: Pool file not found: {pool_path}")
        print("Run:  python generate_config_pool.py")
        sys.exit(1)
    pool_data   = torch.load(pool_path, weights_only=False)
    all_configs = pool_data["configs"]

    start = args.start
    end   = args.end

    if start >= len(all_configs):
        print(f"ERROR: start={start} out of range (pool has {len(all_configs)} configs)")
        sys.exit(1)
    job_configs = all_configs[start:end]

    print(f"Board:  {HARDWARE['name']} | Device: {device}")
    print(f"Configs: [{start}:{end}]  ({len(job_configs)} total)")
    print(f"Epochs: {EPOCHS} (two-step: {max(1,EPOCHS//2)} prune + {EPOCHS-max(1,EPOCHS//2)} quant)"
          f"  |  LR: {LR}  |  Scheduler: CosineAnnealingLR")

    print("Loading MNIST …")
    train_loader, test_loader = get_data_loaders()
    metric_fn        = get_metric()
    calibration_data = next(iter(train_loader))[0].to(device)

    baseline_ckpt = os.path.join(_HERE, "models", "baseline.pth")
    if not os.path.exists(baseline_ckpt):
        print("ERROR: models/baseline.pth not found — run train_baseline.py first.")
        sys.exit(1)
    baseline_model = get_model().to(device)
    baseline_model.load_state_dict(
        torch.load(baseline_ckpt, weights_only=True)["model"]
    )

    nas_data = get_nas_compression_data(
        baseline_model,
        INPUT_SHAPE,
        test_loader,
        metric_fn,
        calibration_data,
        configs=job_configs,
        device=device,
        train=True,
        train_dataloader=train_loader,
        epochs=EPOCHS,
        criterion_fun=nn.CrossEntropyLoss(),
        lr=LR,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY},
        lr_scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": EPOCHS, "eta_min": ETA_MIN},
    )

    out_filename = f"nas_{start}_{end}.pth"
    if args.output_gcs_dir:
        out_path = f"/tmp/{out_filename}"
    else:
        os.makedirs(NAS_DATA_DIR, exist_ok=True)
        out_path = os.path.join(NAS_DATA_DIR, out_filename)

    torch.save({
        "nas_parameters": dict(nas_data),
        "config_start":   start,
        "config_end":     end,
        "n_configs":      len(job_configs),
        "hardware":       HARDWARE["name"],
        "epochs":         EPOCHS,
        "lr":             LR,
    }, out_path)

    n_saved = len(nas_data.get("metric", []))
    print(f"\nSaved {n_saved} samples → {out_path}")

    if args.output_gcs_dir:
        gcs_dest = args.output_gcs_dir.rstrip("/") + "/" + out_filename
        print(f"Uploading results to {gcs_dest} …")
        _gcs_cp(out_path, gcs_dest)
        print("Upload complete.")


if __name__ == "__main__":
    main()
