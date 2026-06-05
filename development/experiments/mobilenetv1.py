"""
MobileNetV1 for CIFAR-100, expressed in DMC Sequential notation.

Architecture follows Howard et al. (2017) "MobileNets: Efficient Convolutional
Neural Networks for Mobile Vision Applications" with CIFAR adaptations:
  - Initial conv uses stride=1 (not stride=2) because inputs are 32×32
  - Three stride-2 downsampling stages: 32→16→8→4
  - AvgPool(4,4) collapses spatial dims before the classifier head
  - Width multiplier α (default 0.5) scales all channel counts uniformly

With α=0.5 the model has ~858 K parameters (3.4 MB FP32), compressible to
≤200 KB for the ATmega2560 (Arduino Mega) target via the DMC pipeline.
"""

from development import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    AvgPool2d,
    Flatten,
    Linear,
)

NUM_CLASSES   = 100
WIDTH_MULT    = 0.5   # α: scales all channel counts


def _c(channels: int, width_mult: float = WIDTH_MULT) -> int:
    """Apply width multiplier and round to nearest int (min 1)."""
    return max(1, int(channels * width_mult))


def _dw_sep_block(in_ch: int, out_ch: int, stride: int = 1) -> list:
    """
    Depthwise-separable convolution block.
      DW 3×3 (groups=in_ch) → BN → ReLU → PW 1×1 → BN → ReLU
    Both convolutions use padding=(1,1) / (0,0) inside DMC's Conv2d directly
    (no separate ConstantPad2d needed for 3×3 depthwise with same-padding).
    """
    return [
        Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
               padding=(1, 1), groups=in_ch, bias=False),
        BatchNorm2d(in_ch),
        ReLU(),
        Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
        BatchNorm2d(out_ch),
        ReLU(),
    ]


def get_model(width_mult: float = WIDTH_MULT, num_classes: int = NUM_CLASSES) -> Sequential:
    """
    Return a randomly initialised MobileNetV1 DMC Sequential model.

    Args:
        width_mult: Channel width multiplier α.  Default 0.5.
            α=0.25 → ~66 K params (~265 KB FP32) — easiest to compress.
            α=0.50 → ~858 K params (~3.4 MB FP32) — good accuracy / size trade-off.
            α=1.00 → ~3.2 M params (~13 MB FP32) — full model, harder to compress.
        num_classes: Number of output classes. Default 100 (CIFAR-100).
    """
    def c(ch: int) -> int:
        return _c(ch, width_mult)

    layers = [
        # ── Stem ─────────────────────────────────────────────────────────────
        Conv2d(3, c(32), kernel_size=3, stride=1, padding=(1, 1), bias=False),
        BatchNorm2d(c(32)),
        ReLU(),

        # ── Stage 1: 32×32, 32→64 ────────────────────────────────────────────
        *_dw_sep_block(c(32), c(64)),

        # ── Stage 2: 32→16, 64→128 ───────────────────────────────────────────
        *_dw_sep_block(c(64), c(128), stride=2),
        *_dw_sep_block(c(128), c(128)),

        # ── Stage 3: 16→8, 128→256 ───────────────────────────────────────────
        *_dw_sep_block(c(128), c(256), stride=2),
        *_dw_sep_block(c(256), c(256)),

        # ── Stage 4: 8→4, 256→512 ────────────────────────────────────────────
        *_dw_sep_block(c(256), c(512), stride=2),
        *_dw_sep_block(c(512), c(512)),
        *_dw_sep_block(c(512), c(512)),
        *_dw_sep_block(c(512), c(512)),
        *_dw_sep_block(c(512), c(512)),
        *_dw_sep_block(c(512), c(512)),

        # ── Stage 5: 4×4, 512→1024 (no additional stride — spatial already 4) ─
        *_dw_sep_block(c(512), c(1024)),
        *_dw_sep_block(c(1024), c(1024)),

        # ── Head ──────────────────────────────────────────────────────────────
        AvgPool2d(kernel_size=(4, 4)),
        Flatten(),
        Linear(c(1024), num_classes, bias=True),
    ]

    return Sequential(*layers)
