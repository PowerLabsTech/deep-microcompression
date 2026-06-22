#!/usr/bin/env python3
"""
MobileNetV1 (x0.5) CIFAR-100 — DMC Comprehensive Experiment Runner

Unlike ResNet-56 / MobileNetV2, MobileNetV1 has no pretrained CIFAR-100 weights,
so this script trains an FP32 baseline from scratch (once, checkpointed) before
sweeping pruning sparsity × quantization bitwidth configurations.

Target board: ATmega2560 (Arduino Mega) — 8 KB SRAM, 256 KB Flash.

Run from the project root:
    python experiments/mobilenetv1_cifar100/run_experiments.py
    python experiments/mobilenetv1_cifar100/run_experiments.py --baseline_epochs 150
"""

import sys
import os
import json
import argparse
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils import data as torch_data
from torchvision import datasets, transforms

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(SCRIPT_DIR)

from development import QuantizationGranularity, QuantizationScheme
from development.experiments.mobilenetv1 import get_model
from development.experiments.cifar100    import get_metric

# ── Reproducibility & device ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (3, 32, 32)

torch.manual_seed(42)
if DEVICE == "cuda":
    torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True

# ── I/O paths ───────────────────────────────────────────────────────────────
LOG_FILE      = os.path.join(SCRIPT_DIR, "experiment_results.json")
CKPT_DIR      = os.path.join(SCRIPT_DIR, "checkpoints")
DATASET_DIR   = os.path.join(PROJECT_ROOT, "../../Datasets/CIFAR_100/")
BASELINE_CKPT = os.path.join(CKPT_DIR, "baseline_pretrained.pth")
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Metrics ─────────────────────────────────────────────────────────────────
TOP1 = get_metric()
TOP5 = lambda yp, yt: (yp.topk(5, dim=1).indices == yt.unsqueeze(1)).any(dim=1).float().mean().item() * 100
METRICS = {"top1acc": TOP1, "top5acc": TOP5}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders with CIFAR-100 augmentation
# ─────────────────────────────────────────────────────────────────────────────
def get_loaders(batch_size: int = 256):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    n_workers = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=n_workers, pin_memory=(DEVICE == "cuda"), persistent_workers=True)
    train_ds = datasets.CIFAR100(DATASET_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(DATASET_DIR, train=False, download=True, transform=test_tf)
    return (
        torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        torch_data.DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def load_results() -> dict:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            r = json.load(f)
        print(f"Resuming — {len(r)} results already logged.")
        return r
    return {}


def save_result(results: dict, label: str, eval_dict: dict,
                size_b: int, workspace_b: int, baseline_b: int | None = None):
    entry = {
        "label":         label,
        "accuracy_top1": round(eval_dict.get("top1acc", 0.0), 4),
        "accuracy_top5": round(eval_dict.get("top5acc", 0.0), 4),
        "size_bytes":    size_b,
        "size_kb":       round(size_b / 1024, 1),
        "workspace_kb":  round(workspace_b / 1024, 1),
        "timestamp":     datetime.now().isoformat(),
    }
    if baseline_b:
        entry["compression_ratio"] = round(baseline_b / size_b, 2)
    results[label] = entry
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    cr_str = f"  |  {baseline_b/size_b:.1f}x" if baseline_b else ""
    print(f"\n{'='*65}\n  DONE ✓  {label}")
    print(f"  Top-1: {entry['accuracy_top1']:.2f}%  Top-5: {entry['accuracy_top5']:.2f}%"
          f"  Size: {size_b/1024:.1f} KB{cr_str}\n{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Compression configs
# ─────────────────────────────────────────────────────────────────────────────
def build_pruning_cfg(model, sparsity: float, keep_stem: bool = True) -> dict:
    """Apply uniform channel sparsity to every prunable group; keep stem dense.

    For MobileNetV1 the prunable groups are the *pointwise* convs (the depthwise
    convs share channel counts and are pruned in lock-step by the framework).
    Built through the model's decoder so group names map to the right modules.
    """
    ss = model.get_compression_possible_hyperparameters()
    groups = [k for k in ss if k.startswith("prune_channel.sparsity.")]
    flat = {full: (0.0 if (keep_stem and i == 0) else sparsity)
            for i, full in enumerate(groups)}
    flat["prune_channel.metric"] = "l2"
    return model.decode_compression_dict_hyperparameter(flat)


def quant_cfg(w_bits: int, a_bits: int) -> dict:
    return {
        "quantize": {
            "scheme":              QuantizationScheme.STATIC,
            "activation_bitwidth": a_bits,
            "parameter_bitwidth":  w_bits,
            "granularity":         QuantizationGranularity.PER_CHANNEL,
        }
    }


W8A8 = quant_cfg(8, 8)
W4A8 = quant_cfg(4, 8)
W4A4 = quant_cfg(4, 4)


# ── LR-scheduler shim ────────────────────────────────────────────────────────
class _CosineWrapper:
    def __init__(self, sched): self._s = sched
    def step(self, loss=None): self._s.step()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (train from scratch — no pretrained MobileNetV1 CIFAR-100 weights)
# ─────────────────────────────────────────────────────────────────────────────
def train_baseline(train_loader, test_loader, n_epochs: int):
    print(f"  Training MobileNetV1 baseline from scratch for {n_epochs} epochs …")
    m = get_model().to(DEVICE)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.SGD(m.parameters(), lr=0.1, weight_decay=4e-5, momentum=0.9)
    sch  = _CosineWrapper(optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs))
    m.fit(train_loader, n_epochs, crit, opt, sch,
          validation_dataloader=test_loader, metrics=METRICS, device=DEVICE)
    torch.save({"model": m.cpu().state_dict()}, BASELINE_CKPT)
    print(f"  Baseline saved → {BASELINE_CKPT}")
    return m.to(DEVICE)


def _load_baseline():
    m = get_model().to(DEVICE)
    ckpt = torch.load(BASELINE_CKPT, weights_only=True, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    m.load_state_dict(state)
    return m.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def finetune(model, train_loader, test_loader, n_epochs: int, lr: float = 1e-2):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.SGD(model.parameters(), lr=lr, weight_decay=4e-5, momentum=0.9)
    sch  = _CosineWrapper(optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs))
    model.fit(train_loader, n_epochs, crit, opt, sch,
              validation_dataloader=test_loader, metrics=METRICS, device=DEVICE)
    return model


def run_pruning(p_cfg, train_loader, test_loader, ckpt_path, n_epochs=30):
    base = _load_baseline()
    if os.path.exists(ckpt_path):
        print(f"  [checkpoint hit] {os.path.basename(ckpt_path)} — skipping prune/finetune")
        m = base.init_compress(p_cfg, INPUT_SHAPE).to(DEVICE)
        m.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=False)
        return m.to(DEVICE)
    print("  Applying structured pruning …")
    m = base.init_compress(p_cfg, INPUT_SHAPE).to(DEVICE)
    finetune(m, train_loader, test_loader, n_epochs=n_epochs, lr=1e-2)
    torch.save(m.cpu().state_dict(), ckpt_path)
    print(f"  Pruned model saved → {ckpt_path}")
    return m.to(DEVICE)


def run_qat(pruned_model, full_cfg, train_loader, test_loader, n_epochs=20):
    calib = next(iter(train_loader))[0].to(DEVICE)
    m = pruned_model.init_compress(full_cfg, INPUT_SHAPE, calib).to(DEVICE)
    finetune(m, train_loader, test_loader, n_epochs=n_epochs, lr=5e-4)
    return m


def measure(model, test_loader):
    model.eval()
    ev = model.evaluate(test_loader, METRICS, device=DEVICE)
    sz = model.fuse(device=DEVICE).get_size_in_bytes()
    ws = model.get_workspace_size(INPUT_SHAPE)
    return ev, sz, ws


# ─────────────────────────────────────────────────────────────────────────────
# Experiment plan  (sparsity, quant_cfg, tag, prune_epochs, qat_epochs)
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    (0.20, W8A8, "W8A8", 30, 15),
    (0.30, W8A8, "W8A8", 30, 15),
    (0.40, W8A8, "W8A8", 30, 15),
    (0.50, W8A8, "W8A8", 30, 15),
    (0.20, W4A8, "W4A8", 30, 20),
    (0.30, W4A8, "W4A8", 30, 20),
    (0.40, W4A8, "W4A8", 30, 20),
    (0.30, W4A4, "W4A4", 30, 25),
    (0.40, W4A4, "W4A4", 30, 25),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_epochs", type=int, default=150,
                        help="Epochs to train the FP32 baseline from scratch (first run only).")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Board  : ATmega2560 (Arduino Mega) — 8 KB SRAM, 256 KB Flash")
    print(f"Log    : {LOG_FILE}\n")

    results = load_results()
    train_loader, test_loader = get_loaders()

    # ── Stage 0: Baseline FP32 (train from scratch if missing) ─────────────
    label = "Baseline (FP32)"
    if not os.path.exists(BASELINE_CKPT):
        m = train_baseline(train_loader, test_loader, n_epochs=args.baseline_epochs)
    if label not in results:
        print(f"\n{'─'*65}\n  {label}\n{'─'*65}")
        m = _load_baseline()
        ev, sz, ws = measure(m, test_loader)
        save_result(results, label, ev, sz, ws)
    baseline_size = results["Baseline (FP32)"]["size_bytes"]

    # ── Stage 1: W8A8 QAT only (no pruning) ────────────────────────────────
    label = "W8A8 QAT Only"
    if label not in results:
        print(f"\n{'─'*65}\n  {label}\n{'─'*65}")
        m = run_qat(_load_baseline(), W8A8, train_loader, test_loader, n_epochs=15)
        ev, sz, ws = measure(m, test_loader)
        save_result(results, label, ev, sz, ws, baseline_size)

    # ── Stages 2+: Pruning × quantisation sweep ────────────────────────────
    for sparsity, q_cfg, q_tag, p_ep, qat_ep in EXPERIMENTS:
        sp_pct = int(sparsity * 100)
        p_cfg  = build_pruning_cfg(get_model(), sparsity)
        ckpt   = os.path.join(CKPT_DIR, f"pruned_{sp_pct}pct.pth")

        fp_label = f"Pruned {sp_pct}% (FP32)"
        if fp_label not in results:
            print(f"\n{'─'*65}\n  {fp_label}\n{'─'*65}")
            m = run_pruning(p_cfg, train_loader, test_loader, ckpt, n_epochs=p_ep)
            ev, sz, ws = measure(m, test_loader)
            save_result(results, fp_label, ev, sz, ws, baseline_size)
        else:
            print(f"  [skip] {fp_label}")

        qat_label = f"Pruned {sp_pct}% + {q_tag}"
        if qat_label not in results:
            print(f"\n{'─'*65}\n  {qat_label}\n{'─'*65}")
            pruned = run_pruning(p_cfg, train_loader, test_loader, ckpt, n_epochs=p_ep)
            m = run_qat(pruned, {**p_cfg, **q_cfg}, train_loader, test_loader, n_epochs=qat_ep)
            ev, sz, ws = measure(m, test_loader)
            save_result(results, qat_label, ev, sz, ws, baseline_size)
        else:
            print(f"  [skip] {qat_label}")

    # ── Final summary table ────────────────────────────────────────────────
    print(f"\n\n{'='*75}\n  EXPERIMENT SUMMARY — MobileNetV1 x0.5 on CIFAR-100 with DMC\n{'='*75}")
    print(f"  {'Configuration':<38} {'Top-1':>7} {'Top-5':>7} {'Size KB':>9} {'Ratio':>7}")
    print(f"  {'─'*70}")
    for v in results.values():
        cr = f"{v.get('compression_ratio', 1.0):.1f}x"
        print(f"  {v['label']:<38} {v['accuracy_top1']:>6.2f}% {v['accuracy_top5']:>6.2f}%"
              f" {v['size_kb']:>8.1f} {cr:>7}")
    print(f"{'='*75}\n\nFull results → {LOG_FILE}")


if __name__ == "__main__":
    main()
