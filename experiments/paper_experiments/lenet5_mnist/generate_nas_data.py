#!/usr/bin/env python3
"""
Generate NAS compression data for LeNet-5/MNIST.
Target board: ATmega328P (Arduino Uno) — 2 KB SRAM, 32 KB Flash.

Usage:
    python generate_config_pool.py --n_configs 1000
    python generate_nas_data.py --start 0 --end 50
    python generate_nas_data.py --start 0 --end 50 \\
        --pool_gcs_uri gs://... --baseline_gcs_uri gs://... --output_gcs_dir gs://...

Fine-tuning HPs (validated via nas_hp_benchmark.py):
    lr=1e-2, CosineAnnealingLR, 5 epochs → 0.53pp mean error vs 20-epoch reference.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../.."))
sys.path.insert(0, os.path.join(_HERE, "../shared"))

from generate_nas_data import main
from lenet5             import get_model
from mnist              import get_data_loaders, get_metric
from hardware_specs     import ATMEGA328P

main(
    exp_dir        = _HERE,
    hardware       = ATMEGA328P,
    input_shape    = (1, 28, 28),
    model_fn       = lambda args: get_model(),
    data_fn        = get_data_loaders,
    metric_fn      = get_metric,
    description    = "LeNet-5/MNIST",
    has_baseline   = True,
    has_eval_subset= False,
    defaults       = {
        "lr": 1e-2, "epochs": 5, "eta_min": 1e-5,
        "momentum": 0.9, "weight_decay": 5e-4
    },
)