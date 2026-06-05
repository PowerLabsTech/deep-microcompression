"""
MobileNetV2 for CIFAR-100, expressed in DMC Sequential notation.
Pretrained weights loaded from chenyaofo/pytorch-cifar-models (x0.5 variant).

CIFAR adaptation vs ImageNet MobileNetV2:
  - Stem conv stride = 1 (not 2)
  - Stage 2 stride  = 1 (not 2) — see chenyaofo source comment
  Spatial dims: 32 → 32 → 32 → 32 → 16 → 8 → 8 → 4 → 4 → AvgPool(4×4) → 1

Width multiplier α = 0.5  → 1,393,956 parameters (~5.3 MB FP32)
Target board: ATmega2560 (Arduino Mega) — 8 KB SRAM, 256 KB Flash
"""

import torch
from development import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU6,
    AvgPool2d,
    Flatten,
    Linear,
    Branch,
    Block,
)

NUM_CLASSES  = 100
WIDTH_MULT   = 0.5

# (expand_ratio, out_channels, n_blocks, stride)
# NOTE: stage 2 stride changed 2→1 for CIFAR (same as chenyaofo source)
_SETTINGS = [
    (1,  16, 1, 1),
    (6,  24, 2, 1),   # stride=1 for CIFAR
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _make_div8(v: float) -> int:
    """Round to nearest multiple of 8 (min 8). Matches chenyaofo's _make_divisible."""
    new_v = max(8, int(v + 4) // 8 * 8)
    if new_v < 0.9 * v:
        new_v += 8
    return new_v


def _hidden(in_ch: int, t: int) -> int:
    """Expanded channel count, matching chenyaofo: int(round(in_ch * t))."""
    return int(round(in_ch * t))


def _block_no_shortcut(in_ch: int, out_ch: int, t: int, stride: int) -> list:
    """
    Inverted residual without skip connection (stride > 1 or in_ch ≠ out_ch).
    Layers are appended directly into the Sequential.
    """
    h = _hidden(in_ch, t)
    if t == 1:
        # No pw-expand when expand_ratio == 1
        return [
            Conv2d(in_ch, h, kernel_size=3, stride=stride, padding=(1, 1), groups=in_ch, bias=False),
            BatchNorm2d(h),
            ReLU6(),
            Conv2d(h, out_ch, kernel_size=1, bias=False),
            BatchNorm2d(out_ch),
        ]
    return [
        Conv2d(in_ch, h, kernel_size=1, bias=False),      # pw-expand
        BatchNorm2d(h),
        ReLU6(),
        Conv2d(h, h, kernel_size=3, stride=stride,        # dw
               padding=(1, 1), groups=h, bias=False),
        BatchNorm2d(h),
        ReLU6(),
        Conv2d(h, out_ch, kernel_size=1, bias=False),     # pw-linear (no activation)
        BatchNorm2d(out_ch),
    ]


def _block_with_shortcut(ch: int, t: int) -> list:
    """
    Inverted residual with identity skip connection (stride=1, in_ch == out_ch).
    Wrapped in Branch so DMC adds Block(x) + x automatically.
    No activation after the Branch (MobileNetV2 design).
    """
    h = _hidden(ch, t)
    if t == 1:
        block = Block(
            Conv2d(ch, h, kernel_size=3, stride=1, padding=(1, 1), groups=ch, bias=False),
            BatchNorm2d(h),
            ReLU6(),
            Conv2d(h, ch, kernel_size=1, bias=False),
            BatchNorm2d(ch),
        )
    else:
        block = Block(
            Conv2d(ch, h, kernel_size=1, bias=False),
            BatchNorm2d(h),
            ReLU6(),
            Conv2d(h, h, kernel_size=3, stride=1, padding=(1, 1), groups=h, bias=False),
            BatchNorm2d(h),
            ReLU6(),
            Conv2d(h, ch, kernel_size=1, bias=False),
            BatchNorm2d(ch),
        )
    return [Branch(block)]


def get_dmc_model(width_mult: float = WIDTH_MULT) -> Sequential:
    """Return a randomly-initialised DMC MobileNetV2 with the given width multiplier."""
    first_ch  = _make_div8(32 * width_mult)     # 16 for α=0.5
    last_ch   = _make_div8(1280 * max(1.0, width_mult))  # 1280 for α≤1.0

    layers = [
        Conv2d(3, first_ch, kernel_size=3, stride=1, padding=(1, 1), bias=False),
        BatchNorm2d(first_ch),
        ReLU6(),
    ]

    in_ch = first_ch
    for t, c, n, s in _SETTINGS:
        out_ch = _make_div8(c * width_mult)
        for i in range(n):
            stride = s if i == 0 else 1
            if stride == 1 and in_ch == out_ch:
                layers += _block_with_shortcut(in_ch, t)
            else:
                layers += _block_no_shortcut(in_ch, out_ch, t, stride)
            in_ch = out_ch

    layers += [
        Conv2d(in_ch, last_ch, kernel_size=1, bias=False),   # last conv
        BatchNorm2d(last_ch),
        ReLU6(),
        AvgPool2d(kernel_size=(4, 4)),   # 4×4 → 1×1 (CIFAR spatial after downsampling)
        Flatten(),
        Linear(last_ch, NUM_CLASSES, bias=True),
    ]

    return Sequential(*layers)


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

def _transfer_weights(hub_model: torch.nn.Module, dmc_model: Sequential) -> Sequential:
    """
    Map hub weights → DMC model by positional matching.

    Both models traverse in the same layer order; we skip num_batches_tracked
    (it has no meaningful value to transfer) and match every other tensor by
    position, verifying shapes before writing.
    """
    SKIP = "num_batches_tracked"
    hub_sd = hub_model.state_dict()
    dmc_sd = dmc_model.state_dict()

    hub_pairs = [(k, v) for k, v in hub_sd.items() if SKIP not in k]
    dmc_pairs = [(k, v) for k, v in dmc_sd.items() if SKIP not in k]

    if len(hub_pairs) != len(dmc_pairs):
        hub_keys = [k for k, _ in hub_pairs]
        dmc_keys = [k for k, _ in dmc_pairs]
        raise ValueError(
            f"Tensor count mismatch: hub={len(hub_pairs)}, dmc={len(dmc_pairs)}\n"
            f"First hub key with no match: "
            f"{hub_keys[len(dmc_keys)] if len(hub_keys) > len(dmc_keys) else dmc_keys[len(hub_keys)]}"
        )

    new_sd = dict(dmc_sd)   # keep num_batches_tracked at default
    for i, ((hub_key, hub_val), (dmc_key, dmc_val)) in enumerate(zip(hub_pairs, dmc_pairs)):
        if hub_val.shape != dmc_val.shape:
            raise ValueError(
                f"Shape mismatch at position {i}:\n"
                f"  hub[{hub_key}] = {tuple(hub_val.shape)}\n"
                f"  dmc[{dmc_key}] = {tuple(dmc_val.shape)}"
            )
        new_sd[dmc_key] = hub_val

    missing, unexpected = dmc_model.load_state_dict(new_sd, strict=False)
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    if missing:
        print(f"  Warning — missing keys: {missing}")

    # Sanity-check: param counts must match exactly
    hub_p = sum(p.numel() for p in hub_model.parameters())
    dmc_p = sum(p.numel() for p in dmc_model.parameters())
    assert hub_p == dmc_p, f"Param count mismatch — hub={hub_p:,}, dmc={dmc_p:,}"
    print(f"  All {len(hub_pairs)} tensors transferred. Param count: {dmc_p:,}")

    return dmc_model


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_model(width_mult: float = WIDTH_MULT) -> Sequential:
    """
    Load pretrained MobileNetV2 x0.5 from chenyaofo/pytorch-cifar-models
    and return as a DMC Sequential model ready for compression.
    """
    variant = "x0_5" if width_mult == 0.5 else f"x{str(width_mult).replace('.', '_')}"
    hub_name = f"cifar{NUM_CLASSES}_mobilenetv2_{variant}"
    print(f"Downloading {hub_name} from chenyaofo/pytorch-cifar-models …")

    hub_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        hub_name,
        pretrained=True,
        verbose=False,
        trust_repo=True,
    )
    hub_model.eval()

    dmc_model = get_dmc_model(width_mult)
    dmc_model  = _transfer_weights(hub_model, dmc_model)
    return dmc_model
