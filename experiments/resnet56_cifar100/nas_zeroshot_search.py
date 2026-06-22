#!/usr/bin/env python3
"""
NAS board search (training-free ranking)  —  ResNet-56 / CIFAR-100
==================================================================

Same experiment as the LeNet version, for ResNet-56 on the ATmega2560
(8 KB SRAM / 256 KB Flash):

  1. SAMPLE board-fitting configs with the library NAS generator
     (`get_nas_compression_data`, train=False) — each is compressed, calibrated
     and scored ZERO-SHOT (no fine-tuning); a geometry filter keeps only those
     that fit the board's SRAM + Flash budget.
  2. RANK by zero-shot accuracy.
  3. TRAIN the top-k fully (2-stage: prune -> QAT) and keep the best.
  4. REPORT best deployable model + Spearman rho(zero-shot, trained) + recall@k.

Note: on CIFAR-100 the board-fitting configs need aggressive pruning, so their
zero-shot accuracy can sit near chance (~1%).  The verdict's rho tells us whether
zero-shot ranking carries any signal at this budget.

Run:
    python experiments/resnet56_cifar100/nas_zeroshot_search.py \
        --n_sample 200 --n_train 5 --prune_epochs 30 --qat_epochs 15
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data as torch_data
from torchvision import datasets, transforms

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

from development.experiments.resnet   import get_model
from development.experiments.cifar100 import get_metric
from development.models.utils         import get_nas_compression_data

# ── Reproducibility & device ────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (3, 32, 32)
torch.manual_seed(42)
if DEVICE == "cuda":
    torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True

# ── Board + recipe (ResNet-56 / CIFAR-100) ──────────────────────────────────
SRAM_DEFAULT, FLASH_DEFAULT = 8192, 262144          # ATmega2560
PRUNE_LR, QAT_LR = 1e-2, 5e-4
WEIGHT_DECAY, LABEL_SMOOTHING = 5e-4, 0.1

# ── I/O paths ───────────────────────────────────────────────────────────────
OUT_FILE      = os.path.join(SCRIPT_DIR, "nas_zeroshot_search_results.json")
DATASET_DIR   = os.path.join(PROJECT_ROOT, "../../Datasets/CIFAR_100/")
BASELINE_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "baseline_pretrained.pth")

# ── Metrics ─────────────────────────────────────────────────────────────────
TOP1 = get_metric()
TOP5 = lambda yp, yt: (yp.topk(5, dim=1).indices == yt.unsqueeze(1)).any(dim=1).float().mean().item() * 100
METRICS = {"top1acc": TOP1, "top5acc": TOP5}


# ─────────────────────────────────────────────────────────────────────────────
# Data  (standard CIFAR-100 augmentation)
# ─────────────────────────────────────────────────────────────────────────────
def get_loaders(batch_size: int = 256):
    norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    nw = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=nw, pin_memory=(DEVICE == "cuda"), persistent_workers=True)
    train_ds = datasets.CIFAR100(DATASET_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(DATASET_DIR, train=False, download=True, transform=test_tf)
    return (
        torch_data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        torch_data.DataLoader(test_ds,  batch_size=256,        shuffle=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_baseline():
    m = get_model().to(DEVICE)
    if BASELINE_CKPT and os.path.exists(BASELINE_CKPT):
        ck = torch.load(BASELINE_CKPT, weights_only=True, map_location=DEVICE)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        m.load_state_dict(state, strict=False)
    return m


def geometry(compressed_model) -> tuple[int, int]:
    """Exact (Flash bytes, SRAM workspace bytes) — needs no training."""
    return (compressed_model.fuse(device=DEVICE).get_size_in_bytes(),
            compressed_model.get_workspace_size(INPUT_SHAPE))


def fits_board(compressed_model, sram: int, flash: int) -> bool:
    f, s = geometry(compressed_model)
    return f <= flash and s <= sram


def config_at(data: dict, index: int) -> dict:
    return {k: v[index] for k, v in data.items() if k != "metric"}


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Final training — 2-stage: prune (FP32) -> quantize (QAT)
# ─────────────────────────────────────────────────────────────────────────────
class _CosineWrapper:
    """CosineAnnealingLR.step() takes no metric; fit() passes the val loss."""
    def __init__(self, sched): self._s = sched
    def step(self, loss=None): self._s.step()


def finetune(model, train_loader, test_loader, epochs: int, lr: float):
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt  = optim.SGD(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
    sch  = _CosineWrapper(optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs))
    model.fit(train_loader, epochs, crit, opt, sch,
              validation_dataloader=test_loader, metrics=METRICS, device=DEVICE)


def train_config(decoded_cfg: dict, train_loader, test_loader,
                 prune_epochs: int, qat_epochs: int) -> dict:
    """Stage A: prune the baseline + FP32 fine-tune.  Stage B: quantize + QAT."""
    base = load_baseline()

    prune_cfg = {"prune_channel": decoded_cfg["prune_channel"]}
    pruned = base.init_compress(prune_cfg, INPUT_SHAPE).to(DEVICE)
    finetune(pruned, train_loader, test_loader, prune_epochs, lr=PRUNE_LR)
    pruned_acc = pruned.eval().evaluate(test_loader, METRICS, device=DEVICE)["top1acc"]

    calib = next(iter(test_loader))[0].to(DEVICE)
    q = pruned.init_compress(decoded_cfg, INPUT_SHAPE, calib).to(DEVICE)
    finetune(q, train_loader, test_loader, qat_epochs, lr=QAT_LR)
    ev = q.eval().evaluate(test_loader, METRICS, device=DEVICE)

    flash, sram = geometry(q)
    return {"pruned_fp32_top1": round(pruned_acc, 3),
            "trained_top1": round(ev["top1acc"], 3),
            "trained_top5": round(ev["top5acc"], 3),
            "flash": flash, "sram": sram}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sram",  type=int, default=SRAM_DEFAULT)
    ap.add_argument("--flash", type=int, default=FLASH_DEFAULT)
    ap.add_argument("--n_sample", type=int, default=200, help="board-fitting configs to collect")
    ap.add_argument("--n_train",  type=int, default=5,   help="top configs to train")
    ap.add_argument("--prune_epochs", type=int, default=30)
    ap.add_argument("--qat_epochs",   type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Device {DEVICE}   Board SRAM<={args.sram}B Flash<={args.flash}B")
    train_loader, test_loader = get_loaders()
    calib = next(iter(train_loader))[0].to(DEVICE)

    # ── 1. Sample board-fitting configs, scored ZERO-SHOT (train=False) ──────
    print(f"\nSampling {args.n_sample} board-fitting configs (zero-shot, no training) ...")
    data = get_nas_compression_data(
        get_model(), INPUT_SHAPE, test_loader, TOP1, calib,
        filter=lambda model, cfg: fits_board(model, args.sram, args.flash),
        device=DEVICE, num_data=args.n_sample, train=False, random_seed=args.seed,
    )

    # ── 2. Rank by zero-shot accuracy (best first) ───────────────────────────
    order = np.argsort(data["metric"])[::-1]
    data = {k: [v[i] for i in order] for k, v in data.items()}
    print(f"  zero-shot range: {min(data['metric']):.2f}% .. {max(data['metric']):.2f}%")

    # ── 3. Train the top-k fully; keep results (resumable) ───────────────────
    results = json.load(open(OUT_FILE)) if os.path.exists(OUT_FILE) else \
        {"board": {"sram": args.sram, "flash": args.flash}, "trained": []}
    done = {t["key"] for t in results["trained"]}

    n_train = min(args.n_train, len(data["metric"]))
    print(f"\nTraining top-{n_train} (prune {args.prune_epochs} / QAT {args.qat_epochs}) ...")
    for idx in range(n_train):
        cfg     = config_at(data, idx)
        decoded = get_model().decode_compression_dict_hyperparameter(cfg)
        key     = json.dumps(cfg, sort_keys=True, default=str)
        if key in done:
            print(f"  [skip] rank {idx} already trained"); continue

        zs = float(data["metric"][idx])
        print(f"\n  === rank {idx}: zero-shot {zs:.2f}% ===")
        m = train_config(decoded, train_loader, test_loader, args.prune_epochs, args.qat_epochs)
        m.update({"key": key, "zs_rank": idx, "zeroshot": round(zs, 3), "config": cfg})
        results["trained"].append(m)
        json.dump(results, open(OUT_FILE, "w"), indent=2, default=str)
        print(f"  zero-shot {zs:.2f}%  ->  pruned-FP32 {m['pruned_fp32_top1']:.2f}%  "
              f"->  trained {m['trained_top1']:.2f}%   SRAM={m['sram']}B Flash={m['flash']}B")

    # ── 4. Verdict: does zero-shot order survive training? ───────────────────
    tr = results["trained"]
    if tr:
        zs = [t["zeroshot"] for t in tr]
        ac = [t["trained_top1"] for t in tr]
        best = max(tr, key=lambda d: d["trained_top1"])
        rank_of_best = sorted(range(len(tr)), key=lambda i: zs[i], reverse=True).index(
            max(range(len(tr)), key=lambda i: ac[i]))
        print(f"\n{'='*64}\n  BEST DEPLOYABLE ResNet-56 (zero-shot search)\n{'='*64}")
        print(f"  Top-1 {best['trained_top1']:.2f}%   SRAM={best['sram']}B  Flash={best['flash']}B")
        print(f"  Spearman rho(zero-shot, trained) = {spearman(zs, ac):+.3f}")
        print(f"  best-trained sat at zero-shot rank {rank_of_best} "
              f"(recall@1={rank_of_best < 1}, @3={rank_of_best < 3})")
        print(f"  -> {OUT_FILE}")


if __name__ == "__main__":
    main()
