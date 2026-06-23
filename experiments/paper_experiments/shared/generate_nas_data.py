#!/usr/bin/env python3
"""
Shared backbone for NAS data generation across all DMC experiments.

Each experiment's generate_nas_data.py calls main() with its specific
model/data/hardware config. Everything else — pool loading, argparse,
GCS up/download, saving — lives here.
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


def main(
    exp_dir: str,
    hardware: dict,
    input_shape: tuple,
    model_fn,
    data_fn,
    metric_fn,
    description: str,
    has_baseline: bool = False,
    has_eval_subset: bool = True,
    extra_args: list = None,
    extra_save_fn=None,
    defaults: dict = None,
):
    """
    Args:
        exp_dir:         Absolute path to the experiment directory (pass __file__).
        hardware:        Hardware spec dict (e.g. ATMEGA2560).
        input_shape:     Model input shape tuple, e.g. (3, 32, 32).
        model_fn:        Callable(args) -> Sequential (architecture only, no weights).
        data_fn:         Callable() -> (train_loader, test_loader).
        metric_fn:       Callable() -> metric function.
        description:     Short label for argparse and print output, e.g. "ResNet-56/CIFAR-100".
        has_baseline:    If True, adds --baseline_gcs_uri arg and loads models/baseline.pth.
        has_eval_subset: If True, adds --eval_subset arg and passes it to get_nas_compression_data.
        extra_args:      List of dicts {name, type, default, help, ...} for extra CLI args.
        extra_save_fn:   Callable(args) -> dict of extra keys to merge into the saved .pth.
        defaults:        Override any of: lr, epochs, eta_min, momentum, weight_decay,
                         eval_subset, pool_file.
    """
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
    sys.path.insert(0, _PROJ_ROOT)
    sys.path.insert(0, _HERE)

    from development import get_nas_compression_data

    defaults = defaults or {}
    LR           = defaults.get("lr",           1e-2)
    EPOCHS       = defaults.get("epochs",       5)
    ETA_MIN      = defaults.get("eta_min",      1e-5)
    MOMENTUM     = defaults.get("momentum",     0.9)
    WEIGHT_DECAY = defaults.get("weight_decay", 5e-4)

    parser = argparse.ArgumentParser(description=f"NAS data generation: {description}")
    parser.add_argument("--start",          type=int, required=True,
                        help="First config index to process (inclusive)")
    parser.add_argument("--end",            type=int, required=True,
                        help="Last config index to process (exclusive)")
    parser.add_argument("--pool_file",      type=str,
                        default=defaults.get("pool_file", "config_pool.pth"))
    parser.add_argument("--pool_gcs_uri",   type=str, default=None,
                        help="GCS URI to download the pool file from (cloud runs)")
    parser.add_argument("--output_gcs_dir", type=str, default=None,
                        help="GCS URI prefix to upload results to after completion")
    if has_baseline:
        parser.add_argument("--baseline_gcs_uri", type=str, default=None,
                            help="GCS URI to download baseline.pth from (cloud runs)")
    if has_eval_subset:
        parser.add_argument("--eval_subset", type=int,
                            default=defaults.get("eval_subset", 1000),
                            help="Number of test samples for evaluation (subset for speed)")
    for arg_spec in (extra_args or []):
        spec = arg_spec.copy()
        name = spec.pop("name")
        parser.add_argument(name, **spec)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Pool ---
    if args.pool_gcs_uri:
        pool_path = f"/tmp/{os.path.basename(args.pool_gcs_uri)}"
        print(f"Downloading pool from {args.pool_gcs_uri} …")
        _gcs_cp(args.pool_gcs_uri, pool_path)
    else:
        pool_path = os.path.join(exp_dir, args.pool_file)

    if not os.path.exists(pool_path):
        print(f"ERROR: Pool file not found: {pool_path}")
        print("Run:  python generate_config_pool.py --n_configs 1000")
        sys.exit(1)

    pool_data   = torch.load(pool_path, weights_only=False)
    all_configs = pool_data["configs"]
    start, end  = args.start, args.end
    if start >= len(all_configs):
        print(f"ERROR: start={start} out of range (pool has {len(all_configs)} configs)")
        sys.exit(1)
    job_configs = all_configs[start:end]

    eval_info = f"  |  Eval subset: {args.eval_subset}" if has_eval_subset else ""
    print(f"Board:   {hardware['name']} | Device: {device}")
    print(f"Configs: [{start}:{end}]  ({len(job_configs)} total)")
    print(f"Epochs:  {EPOCHS} (two-step: {max(1,EPOCHS//2)} prune + {EPOCHS-max(1,EPOCHS//2)} quant)"
          f"  |  LR: {LR}  |  Scheduler: CosineAnnealingLR{eval_info}")

    # --- Data ---
    train_loader, test_loader = data_fn()
    metric           = metric_fn()
    calibration_data = next(iter(train_loader))[0].to(device)

    # --- Baseline (optional) ---
    if has_baseline:
        if args.baseline_gcs_uri:
            baseline_ckpt = "/tmp/baseline.pth"
            print(f"Downloading baseline from {args.baseline_gcs_uri} …")
            _gcs_cp(args.baseline_gcs_uri, baseline_ckpt)
        else:
            baseline_ckpt = os.path.join(exp_dir, "models", "baseline.pth")
        if not os.path.exists(baseline_ckpt):
            print(f"ERROR: baseline.pth not found at {baseline_ckpt}")
            sys.exit(1)

    # --- Model ---
    model = model_fn(args).to(device)
    if has_baseline:
        model.load_state_dict(torch.load(baseline_ckpt, weights_only=True)["model"])

    # --- NAS ---
    nas_kwargs = {}
    if has_eval_subset:
        nas_kwargs["eval_subset_size"] = args.eval_subset

    nas_data = get_nas_compression_data(
        model, input_shape, test_loader, metric, calibration_data,
        configs=job_configs, device=device, train=True,
        train_dataloader=train_loader, epochs=EPOCHS,
        criterion_fun=nn.CrossEntropyLoss(), lr=LR,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY},
        lr_scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs={"T_max": EPOCHS, "eta_min": ETA_MIN},
        **nas_kwargs,
    )

    # --- Save ---
    out_filename = f"nas_{start}_{end}.pth"
    if args.output_gcs_dir:
        out_path = f"/tmp/{out_filename}"
    else:
        nas_data_dir = os.path.join(exp_dir, "nas_data")
        os.makedirs(nas_data_dir, exist_ok=True)
        out_path = os.path.join(nas_data_dir, out_filename)

    save_data = {
        "nas_parameters": dict(nas_data),
        "config_start":   start,
        "config_end":     end,
        "n_configs":      len(job_configs),
        "hardware":       hardware["name"],
        "epochs":         EPOCHS,
        "lr":             LR,
    }
    if extra_save_fn is not None:
        save_data.update(extra_save_fn(args))

    torch.save(save_data, out_path)
    n_saved = len(nas_data.get("metric", []))
    print(f"\nSaved {n_saved} samples → {out_path}")

    if args.output_gcs_dir:
        gcs_dest = args.output_gcs_dir.rstrip("/") + "/" + out_filename
        print(f"Uploading results to {gcs_dest} …")
        _gcs_cp(out_path, gcs_dest)
        print("Upload complete.")
