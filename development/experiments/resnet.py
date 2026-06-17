import torch
from torch import nn

from ..models.utils import convert_from_sequential_torch_to_dmc


import os

from .. import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    Branch,
    Block,
    ReLU,
    AvgPool2d,
    Flatten,
    Linear,
    ConstantPad2d,
)


NUM_CLASSES = 100

HUB_TO_DMC = {
    # Stem
    "conv1":  "conv2d_0",
    "bn1":    "batchnorm2d_0",

    # layer1: 9 identity blocks -> branch_0 .. branch_8
    **{f"layer1.{b}.conv1": f"branch_{b}.sublayer1.conv2d_0"       for b in range(9)},
    **{f"layer1.{b}.bn1":   f"branch_{b}.sublayer1.batchnorm2d_0"  for b in range(9)},
    **{f"layer1.{b}.conv2": f"branch_{b}.sublayer1.conv2d_1"       for b in range(9)},
    **{f"layer1.{b}.bn2":   f"branch_{b}.sublayer1.batchnorm2d_1"  for b in range(9)},

    # layer2.0: downsample block -> branch_9
    "layer2.0.conv1":          "branch_9.sublayer1.conv2d_0",
    "layer2.0.bn1":            "branch_9.sublayer1.batchnorm2d_0",
    "layer2.0.conv2":          "branch_9.sublayer1.conv2d_1",
    "layer2.0.bn2":            "branch_9.sublayer1.batchnorm2d_1",
    "layer2.0.downsample.0":   "branch_9.sublayer2.conv2d_0",
    "layer2.0.downsample.1":   "branch_9.sublayer2.batchnorm2d_0",

    # layer2 blocks 1-8 -> branch_10 .. branch_17
    **{f"layer2.{b}.conv1": f"branch_{9+b}.sublayer1.conv2d_0"      for b in range(1, 9)},
    **{f"layer2.{b}.bn1":   f"branch_{9+b}.sublayer1.batchnorm2d_0" for b in range(1, 9)},
    **{f"layer2.{b}.conv2": f"branch_{9+b}.sublayer1.conv2d_1"      for b in range(1, 9)},
    **{f"layer2.{b}.bn2":   f"branch_{9+b}.sublayer1.batchnorm2d_1" for b in range(1, 9)},

    # layer3.0: downsample block -> branch_18
    "layer3.0.conv1":          "branch_18.sublayer1.conv2d_0",
    "layer3.0.bn1":            "branch_18.sublayer1.batchnorm2d_0",
    "layer3.0.conv2":          "branch_18.sublayer1.conv2d_1",
    "layer3.0.bn2":            "branch_18.sublayer1.batchnorm2d_1",
    "layer3.0.downsample.0":   "branch_18.sublayer2.conv2d_0",
    "layer3.0.downsample.1":   "branch_18.sublayer2.batchnorm2d_0",

    # layer3 blocks 1-8 -> branch_19 .. branch_26
    **{f"layer3.{b}.conv1": f"branch_{18+b}.sublayer1.conv2d_0"      for b in range(1, 9)},
    **{f"layer3.{b}.bn1":   f"branch_{18+b}.sublayer1.batchnorm2d_0" for b in range(1, 9)},
    **{f"layer3.{b}.conv2": f"branch_{18+b}.sublayer1.conv2d_1"      for b in range(1, 9)},
    **{f"layer3.{b}.bn2":   f"branch_{18+b}.sublayer1.batchnorm2d_1" for b in range(1, 9)},

    # Head
    "fc": "linear_0",
}



def identity_block(ch, stride=1):
    """BasicBlock with identity shortcut (same spatial dims and channels)."""
    return [
        Branch(
            Block(
                Conv2d(in_channels=ch,  out_channels=ch, kernel_size=3,
                       stride=stride, padding=(1,1), bias=False),
                BatchNorm2d(num_features=ch),
                ReLU(),
                Conv2d(in_channels=ch, out_channels=ch, kernel_size=3,
                       stride=1, padding=(1,1), bias=False),
                BatchNorm2d(num_features=ch),
            )
        ),
        ReLU(),
    ]


def downsample_block(in_ch, out_ch, stride=2):
    """BasicBlock with learnable 1x1 downsample shortcut."""
    return [
        Branch(
            Block(
                Conv2d(in_channels=in_ch,  out_channels=out_ch, kernel_size=3,
                       stride=stride, padding=(1,1), bias=False),
                BatchNorm2d(num_features=out_ch),
                ReLU(),
                Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                       stride=1, padding=(1,1), bias=False),
                BatchNorm2d(num_features=out_ch),
            ),
            Block(
                Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                       stride=stride, bias=False),
                BatchNorm2d(num_features=out_ch),
            ),
        ),
        ReLU(),
    ]


def get_dmc_model():
    """
    ResNet-56 for CIFAR-10 expressed in DMC Branch/Block notation.

    Architecture (He et al. 2016 Sec 4.2, chenyaofo variant):
      stem:   Conv(3->16, 3x3) + BN + ReLU
      layer1: 9 x identity BasicBlock(16->16)
      layer2: 1 x downsample BasicBlock(16->32, stride=2)
               + 8 x identity BasicBlock(32->32)
      layer3: 1 x downsample BasicBlock(32->64, stride=2)
               + 8 x identity BasicBlock(64->64)
      head:   AdaptiveAvgPool2d(1,1) + Flatten + Linear(64->10)
    """
    layers = []

    # Stem
    layers += [
        Conv2d(in_channels=3, out_channels=16, kernel_size=3,
               stride=1, padding=(1,1), bias=False),
        BatchNorm2d(num_features=16),
        ReLU(),
    ]

    # layer1: 9 identity blocks (16->16)
    for _ in range(9):
        layers += identity_block(16)

    # layer2: 1 downsample + 8 identity (16->32 then 32->32)
    layers += downsample_block(16, 32, stride=2)
    for _ in range(8):
        layers += identity_block(32)

    # layer3: 1 downsample + 8 identity (32->64 then 64->64)
    layers += downsample_block(32, 64, stride=2)
    for _ in range(8):
        layers += identity_block(64)

    # Head
    layers += [
        AvgPool2d(kernel_size=(8, 8)),
        Flatten(),
        Linear(in_features=64, out_features=NUM_CLASSES, bias=True),
    ]

    return Sequential(*layers)

def build_hub_to_dmc_map(dmc_model):
    """
    Build a mapping from hub state_dict keys to DMC state_dict keys.

    We walk the hub's named structure in definition order and match
    it to the DMC names_layers() output in the same order.

    Returns: dict { hub_key: dmc_key }
    """
    # Collect all DMC (conv2d_N, bn_N, linear_N) layer names in order
    dmc_names = [(name, module) for name, module in dmc_model.names_layers()
                 if any(t in name for t in ("conv2d", "batchnorm", "linear"))]
    # print(dmc_model)
    # print(dmc_model.names_layers())

    # Build ordered list of hub param groups:
    # Each entry is (hub_prefix, param_type) in the same traversal order
    # as the DMC model.
    hub_groups = []

    # Stem
    hub_groups.append("conv1")    # Conv2d
    hub_groups.append("bn1")      # BN

    # layer1: 9 identity blocks (no downsample)
    for b in range(9):
        hub_groups.append(f"layer1.{b}.conv1")
        hub_groups.append(f"layer1.{b}.bn1")
        hub_groups.append(f"layer1.{b}.conv2")
        hub_groups.append(f"layer1.{b}.bn2")

    # layer2: block 0 has downsample
    hub_groups.append("layer2.0.conv1")
    hub_groups.append("layer2.0.bn1")
    hub_groups.append("layer2.0.conv2")
    hub_groups.append("layer2.0.bn2")
    hub_groups.append("layer2.0.downsample.0")  # skip Conv2d
    hub_groups.append("layer2.0.downsample.1")  # skip BN
    for b in range(1, 9):
        hub_groups.append(f"layer2.{b}.conv1")
        hub_groups.append(f"layer2.{b}.bn1")
        hub_groups.append(f"layer2.{b}.conv2")
        hub_groups.append(f"layer2.{b}.bn2")

    # layer3: block 0 has downsample
    hub_groups.append("layer3.0.conv1")
    hub_groups.append("layer3.0.bn1")
    hub_groups.append("layer3.0.conv2")
    hub_groups.append("layer3.0.bn2")
    hub_groups.append("layer3.0.downsample.0")
    hub_groups.append("layer3.0.downsample.1")
    for b in range(1, 9):
        hub_groups.append(f"layer3.{b}.conv1")
        hub_groups.append(f"layer3.{b}.bn1")
        hub_groups.append(f"layer3.{b}.conv2")
        hub_groups.append(f"layer3.{b}.bn2")

    # FC head
    hub_groups.append("fc")

    # Verify lengths match
    assert len(hub_groups) == len(dmc_names), (
        f"Mismatch: {len(hub_groups)} hub groups vs {len(dmc_names)} DMC layers.\n"
        f"DMC layers: {[n for n,_ in dmc_names]}\n"
        f"Hub groups: {hub_groups}"
    )

    # Build the key remap
    suffixes = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
    key_map = {}
    for (dmc_name, _), hub_prefix in zip(dmc_names, hub_groups):
        for suffix in suffixes:
            key_map[f"{hub_prefix}.{suffix}"] = f"{dmc_name}.{suffix}"

    return key_map


def load_pretrained_into_dmc(dmc_model):
    """Download hub model and transfer weights into DMC Sequential."""
    print(f"Downloading pretrained cifar{NUM_CLASSES}_resnet56 from chenyaofo/pytorch-cifar-models...")
    hub_model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        f"cifar{NUM_CLASSES}_resnet56",
        pretrained=True,
        verbose=False,
        trust_repo=True
    )
    hub_sd = hub_model.state_dict()

    # Remap: for every hub param 'prefix.suffix', look up prefix in HUB_TO_DMC
    # and emit 'dmc_prefix.suffix'
    suffixes = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]
    remapped = {}
    for hub_prefix, dmc_prefix in HUB_TO_DMC.items():
        for suffix in suffixes:
            hub_key = f"{hub_prefix}.{suffix}"
            if hub_key in hub_sd:
                remapped[f"{dmc_prefix}.{suffix}"] = hub_sd[hub_key]

    missing, unexpected = dmc_model.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"  Unexpected keys (ignored): {unexpected}")
    if missing:
        print(f"  Warning - missing keys: {missing}")
    else:
        print(f"  All {len(remapped)} tensors transferred successfully.")

    # Sanity check: param counts must match exactly
    hub_params  = sum(p.numel() for p in hub_model.parameters())
    dmc_params  = sum(p.numel() for p in dmc_model.parameters())
    assert hub_params == dmc_params, (
        f"Param count mismatch! hub={hub_params}, dmc={dmc_params}"
    )
    print(f"  Param count check passed: {dmc_params:,} params in both models.")
    return dmc_model


def get_model():
    print("\n--- STAGE 1: Loading Pretrained Baseline ---")
    dmc_model = get_dmc_model()
    dmc_model = load_pretrained_into_dmc(dmc_model)
    # print(f"Saving remapped weights to {DMC_BASELINE_FILE}...")
    # torch.save(dmc_model.cpu().state_dict(), DMC_BASELINE_FILE)
    return dmc_model