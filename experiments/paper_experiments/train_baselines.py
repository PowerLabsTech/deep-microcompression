#!/usr/bin/env python3
"""
Train baseline checkpoints for models that have no pretrained hub weights.
Must be run before generate_nas_data.py for:
  - LeNet-5 / MNIST
  - MobileNetV1 (α=0.5) / CIFAR-100

Models with hub pretrained weights (ResNet-56, MobileNetV2, VGG-13) are
handled automatically by get_model() and need no checkpoint here.

Usage:
    python train_baselines.py                        # train all
    python train_baselines.py --models lenet5        # single model
    python train_baselines.py --models lenet5 mobilenetv1
"""
import argparse
import os
import sys

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
sys.path.insert(0, _PROJ_ROOT)

from development import EarlyStopper
from development.experiments.lenet5      import get_model as lenet5_model
from development.experiments.mnist       import get_data_loaders as mnist_loaders, get_metric as mnist_metric
from development.experiments.mobilenetv1 import get_model as mobilenetv1_model
from development.experiments.cifar100    import get_metric as cifar_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# LeNet-5 / MNIST
# ---------------------------------------------------------------------------

def train_lenet5(epochs: int = 30):
    out_dir = os.path.join(_HERE, "lenet5_mnist", "models")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "baseline.pth")

    if os.path.exists(ckpt):
        print(f"[lenet5] Baseline already exists at {ckpt} — skipping.")
        return

    print(f"[lenet5] Training on MNIST for up to {epochs} epochs on {DEVICE} …")
    train_loader, test_loader = mnist_loaders()
    metric   = mnist_metric()
    model    = lenet5_model().to(DEVICE)
    stopper  = EarlyStopper(
        monitor_metric="validation_loss", delta=1e-7,
        mode="min", patience=5, restore_best_state_dict=True,
    )
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    model.fit(
        train_loader, epochs, crit, opt,
        validation_dataloader=test_loader,
        metrics={"acc": metric},
        callbacks=[stopper],
        device=DEVICE,
    )

    acc = model.evaluate(test_loader, {"acc": metric}, device=DEVICE)["acc"]
    print(f"[lenet5] Baseline accuracy: {acc:.2f}%")
    torch.save({"model": model.state_dict(), "accuracy": acc}, ckpt)
    print(f"[lenet5] Saved → {ckpt}")


# ---------------------------------------------------------------------------
# MobileNetV1 / CIFAR-100
# ---------------------------------------------------------------------------

def _cifar100_loaders(dataset_dir: str, batch_size: int = 256):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    n_workers = min(4, os.cpu_count() or 1)
    kw = dict(num_workers=n_workers, pin_memory=(DEVICE == "cuda"), persistent_workers=True)
    train_ds = datasets.CIFAR100(dataset_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(dataset_dir, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(test_ds,  batch_size=256,        shuffle=False, **kw),
    )


def train_mobilenetv1(epochs: int = 150, width_mult: float = 0.5):
    out_dir = os.path.join(_HERE, "mobilenetv1_cifar100", "models")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "baseline.pth")

    if os.path.exists(ckpt):
        print(f"[mobilenetv1] Baseline already exists at {ckpt} — skipping.")
        return

    dataset_dir = os.path.join(_PROJ_ROOT, "Datasets", "CIFAR_100")
    print(f"[mobilenetv1] Training on CIFAR-100 for up to {epochs} epochs on {DEVICE} …")
    train_loader, test_loader = _cifar100_loaders(dataset_dir)
    metric = cifar_metric()
    model  = mobilenetv1_model(width_mult=width_mult).to(DEVICE)
    stopper = EarlyStopper(
        monitor_metric="validation_loss", delta=1e-7,
        mode="min", patience=15, restore_best_state_dict=True,
    )
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    model.fit(
        train_loader, epochs, crit, opt, sched,
        validation_dataloader=test_loader,
        metrics={"acc": metric},
        callbacks=[stopper],
        device=DEVICE,
    )

    acc = model.evaluate(test_loader, {"acc": metric}, device=DEVICE)["acc"]
    print(f"[mobilenetv1] Baseline accuracy: {acc:.2f}%")
    torch.save({"model": model.state_dict(), "accuracy": acc, "width_mult": width_mult}, ckpt)
    print(f"[mobilenetv1] Saved → {ckpt}")


# ---------------------------------------------------------------------------

TRAINERS = {
    "lenet5":      train_lenet5,
    "mobilenetv1": train_mobilenetv1,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        choices=list(TRAINERS), default=list(TRAINERS),
        help="Which baselines to train (default: all)",
    )
    parser.add_argument("--lenet5_epochs",      type=int, default=30)
    parser.add_argument("--mobilenetv1_epochs",  type=int, default=150)
    parser.add_argument("--mobilenetv1_width",   type=float, default=0.5)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if "lenet5" in args.models:
        train_lenet5(epochs=args.lenet5_epochs)

    if "mobilenetv1" in args.models:
        train_mobilenetv1(epochs=args.mobilenetv1_epochs, width_mult=args.mobilenetv1_width)

    print("\nAll requested baselines done.")
