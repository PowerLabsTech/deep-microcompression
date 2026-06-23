#!/usr/bin/env python3
"""
Shared NAS config pool generator for all DMC paper experiments.

Can be run directly (pass --model) or via each experiment's thin stub
(which pre-sets model, out_dir, and baseline defaults).

Direct usage:
    python generate_config_pool.py --model vgg13
    python generate_config_pool.py --model lenet5     --baseline ../lenet5_mnist/models/baseline.pth --out_dir ../lenet5_mnist/
    python generate_config_pool.py --model mobilenetv1 --baseline ../mobilenetv1_cifar100/models/baseline.pth --out_dir ../mobilenetv1_cifar100/
    python generate_config_pool.py --model mobilenetv2 --out_dir ../mobilenetv2_cifar100/
    python generate_config_pool.py --model resnet56   --out_dir ../resnet56_cifar100/

Via stub (from any experiment dir):
    python generate_config_pool.py               # 1000 configs, seed 42
    python generate_config_pool.py --n_configs 2000 --seed 7
"""
import argparse
import importlib
import os
import sys
import time

import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from development import sample_nas_compression_configs
from hardware_specs import ATMEGA328P, ATMEGA2560, make_pool_filter

_REGISTRY = {
    "lenet5": {
        "data_module":    "mnist",
        "model_module":   "lenet5",
        "hardware":       ATMEGA328P,
        "input_shape":    (1, 28, 28),
        "needs_baseline": True,
    },
    "vgg13": {
        "data_module":    "cifar100",
        "model_module":   "vgg13",
        "hardware":       ATMEGA2560,
        "input_shape":    (3, 32, 32),
        "needs_baseline": False,
    },
    "mobilenetv1": {
        "data_module":    "cifar100",
        "model_module":   "mobilenetv1",
        "hardware":       ATMEGA2560,
        "input_shape":    (3, 32, 32),
        "needs_baseline": True,
    },
    "mobilenetv2": {
        "data_module":    "cifar100",
        "model_module":   "mobilenetv2",
        "hardware":       ATMEGA2560,
        "input_shape":    (3, 32, 32),
        "needs_baseline": False,
    },
    "resnet56": {
        "data_module":    "cifar100",
        "model_module":   "resnet",
        "hardware":       ATMEGA2560,
        "input_shape":    (3, 32, 32),
        "needs_baseline": False,
    },
}


def main(defaults=None):
    parser = argparse.ArgumentParser(description="Generate NAS config pool")
    parser.add_argument("--model",        type=str, default=None, choices=list(_REGISTRY),
                        help="Which model to generate configs for")
    parser.add_argument("--n_configs",    type=int, default=1000)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--out",          type=str, default="config_pool.pth",
                        help="Output filename")
    parser.add_argument("--out_dir",      type=str, default=os.getcwd(),
                        help="Directory to write the pool file into (default: cwd)")
    parser.add_argument("--max_attempts", type=int, default=200_000)
    parser.add_argument("--baseline",     type=str, default=None,
                        help="Path to baseline.pth (required for lenet5 and mobilenetv1)")
    parser.add_argument("--width_mult",   type=float, default=0.5,
                        help="Width multiplier (mobilenetv1 only)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="The device to put the model for the config generation")
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()

    if args.model is None:
        parser.error("--model is required. Choices: " + ", ".join(_REGISTRY))

    cfg         = _REGISTRY[args.model]
    hardware    = cfg["hardware"]
    input_shape = cfg["input_shape"]

    suffix = f" | α={args.width_mult}" if args.model == "mobilenetv1" else ""
    print(f"Model: {args.model} | Seed: {args.seed} | Device: {args.device} | Target: {args.n_configs} configs{suffix}")

    data_mod         = importlib.import_module(cfg["data_module"])
    train_loader, _  = data_mod.get_data_loaders()
    calibration_data = next(iter(train_loader))[0].to(args.device)

    model_mod = importlib.import_module(cfg["model_module"])
    if cfg["needs_baseline"]:
        baseline_path = args.baseline
        if not baseline_path or not os.path.exists(baseline_path):
            print(f"ERROR: baseline.pth not found at: {baseline_path}")
            print("Pass --baseline <path> or run reproduce.ipynb first.")
            sys.exit(1)
        kwargs = {"width_mult": args.width_mult} if args.model == "mobilenetv1" else {}
        baseline_model = model_mod.get_model(**kwargs).to(args.device)
        baseline_model.load_state_dict(torch.load(baseline_path, weights_only=True)["model"])
    else:
        print(f"Loading {args.model} pretrained weights …")
        baseline_model = model_mod.get_model().to(args.device)

    nas_filter = make_pool_filter(hardware, input_shape)

    print("Sampling unique valid configs …")
    start_time = time.time()
    pool, attempts = sample_nas_compression_configs(
        baseline_model, input_shape, calibration_data,
        n_configs=args.n_configs,
        filter=nas_filter,
        device=args.device,
        random_seed=args.seed,
        deduplicate=True,
        max_attempts=args.max_attempts,
    )
    elapsed = time.time() - start_time
    print(f"Generation took {elapsed:.1f}s | {attempts} attempts for {len(pool)} configs")

    save_data = {
        "configs":     pool,
        "n_configs":   len(pool),
        "seed":        args.seed,
        "hardware":    hardware["name"],
        "input_shape": input_shape,
        "head":        0
    }
    if args.model == "mobilenetv1":
        save_data["width_mult"] = args.width_mult

    os.makedirs(args.out_dir, exist_ok=True)
    out_path  = os.path.join(args.out_dir, args.out)

    torch.save(save_data, out_path)

    print(f"\nSaved {len(pool)} unique configs → {out_path}")
    print(f"\nUsage with 20 jobs of 50 configs each:")
    print(f"  python generate_nas_data.py --start 0 --end 50 --pool_file {args.out}")
    print(f"  ...  (up to --start {len(pool) - 50} --end {len(pool)})")


if __name__ == "__main__":
    main()
