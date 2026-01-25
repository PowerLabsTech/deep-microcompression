"""
@file fuse.py
@brief Layer Fusion Utilities for Bare-Metal Optimization

It merges separate layers (Conv2d, BatchNorm, ReLU) into unified "Fused" layers.

Role in DMC Pipeline:
1.  Pre-Quantization: BatchNorm folding is required before calibration to 
    ensure integer scales capture the true effective weight distribution.
2.  Inference Optimization: Reduces the number of function calls and 
    intermediate SRAM buffers required by the generated C library.
"""
import torch

from ..layers.conv import Conv2d
from ..layers.batchnorm import BatchNorm2d
from ..layers.linear import Linear
from ..layers.activation import ReLU, ReLU6
from ..layers.fused_layers import LinearReLU, Conv2dReLU, LinearReLU6, Conv2dReLU6


@torch.no_grad()
def init_dmc_parameter(original_layer, fused_layer):
    """
    Transfers DMC Pipeline State from source layers to the new fused layer.

    When layers are fused (e.g., Conv+BN -> Conv), the new object must inherit:
    1.  Pruning Masks: `weight_prune_channel`, `is_pruned_channel`.
    2.  Quantization Observers: `input_quantize`, `output_quantize`.

    This ensures that optimization decisions made in earlier pipeline stages 
    are not lost during the fusion graph transformation.
    """
    if "_dmc" in original_layer.__dict__:
        fused_layer.__dict__["_dmc"] = original_layer.__dict__["_dmc"]

    if hasattr(original_layer, "weight_prune_channel"):
        fused_layer.weight_prune_channel = original_layer.weight_prune_channel

    if hasattr(original_layer, "bias_prune_channel"):
        fused_layer.bias_prune_channel = original_layer.bias_prune_channel

    if hasattr(original_layer, "is_pruned_channel"):
        fused_layer.is_pruned_channel = original_layer.is_pruned_channel

    if hasattr(original_layer, "weight_quantize"):
        fused_layer.weight_quantize = original_layer.weight_quantize

    if hasattr(original_layer, "bias_quantize"):
        fused_layer.bias_quantize = original_layer.bias_quantize
        
    if hasattr(original_layer, "input_quantize"):
        fused_layer.input_quantize = original_layer.input_quantize

    if hasattr(original_layer, "output_quantize"):
        fused_layer.output_quantize = original_layer.output_quantize

    if hasattr(original_layer, "is_quantized"):
        fused_layer.is_quantized = original_layer.is_quantized

    return
        
@torch.no_grad()
def fuse_conv2d_batchnorm2d(conv2d, batchnorm2d):
    """
    Folds BatchNorm2d into Conv2d weights.

    Mathematically transforms:
        y = (W*x + b - mean) * (gamma / sigma) + beta
    Into:
        y = W' * x + b'
        Where W' = W * (gamma / sigma)
        And   b' = (b - mean) * (gamma / sigma) + beta

    Why DMC needs this:
    Running a standalone BatchNorm layer 
    on a microcontroller is inefficient and breaks the integer-only flow. 
    Folding bakes the normalization constants into the weights *before* they 
    are quantized to int8.

    Args:
        conv2d: Source convolution layer.
        batchnorm2d: Source batchnorm layer (must follow conv2d).

    Returns:
        A standard Conv2d layer containing the fused weights and bias.
    """
    assert isinstance(conv2d, Conv2d) and isinstance(batchnorm2d, BatchNorm2d), "conv2d has to be of Conv2d type and batchnorm2d has to be BatchNorm2d type"
    assert conv2d.out_channels == batchnorm2d.num_features, f"conv2d and batchnorm not fuseable, conv2d has {conv2d.out_channels} out_channels and batchnorm2d has {batchnorm2d.num_features} num_features, the must tbe equal"
    fused_layer = Conv2d(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = True
    )
    fused_layer.weight.copy_(conv2d.weight * batchnorm2d.folded_weight.view(-1,1,1,1))
    if conv2d.bias is not None:
        fused_layer.bias.copy_(conv2d.bias * batchnorm2d.folded_weight + batchnorm2d.folded_bias) # type: ignore
    else:
        fused_layer.bias.copy_(batchnorm2d.folded_bias) # type: ignore

    return fused_layer


@torch.no_grad()
def fuse_linear_relu(linear, relu):
    """Fuses Linear and ReLU for optimized C-code generation."""
    assert isinstance(linear, Linear) and isinstance(relu, ReLU), "linear has to be of Linear type and relu has to ReLU type"
    fused_layer = LinearReLU(
        out_features = linear.out_features,
        in_features = linear.in_features,
        bias = linear.bias is not None
    )
    fused_layer.weight.copy_(linear.weight)
    if linear.bias is not None:
        fused_layer.bias.copy_(linear.bias)
    return fused_layer


@torch.no_grad()
def fuse_linear_relu6(linear, relu6):
    """Fuses Linear and ReLU6 (common in quantized mobile models)."""
    assert isinstance(linear, Linear) and isinstance(relu6, ReLU6), "linear has to be of Linear type and relu6 has to ReLU6 type"
    fused_layer = LinearReLU6(
        out_features = linear.out_features,
        in_features = linear.in_features,
        bias = linear.bias is not None
    )
    fused_layer.weight.copy_(linear.weight)
    if linear.bias is not None:
        fused_layer.bias.copy_(linear.bias)
    return fused_layer


@torch.no_grad()
def fuse_conv2d_relu(conv2d, relu):
    """Fuses Conv2d and ReLU."""
    assert isinstance(conv2d, Conv2d) and isinstance(relu, ReLU), "conv2d has to be of Conv2d type and relu has to ReLU type"
    fused_layer = Conv2dReLU(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = conv2d.bias is not None
    )
    fused_layer.weight.copy_(conv2d.weight)
    if conv2d.bias is not None and fused_layer.bias is not None:
        fused_layer.bias.copy_(conv2d.bias)
    return fused_layer




@torch.no_grad()
def fuse_conv2d_relu6(conv2d, relu6):
    """Fuses Conv2d and ReLU6."""
    assert isinstance(conv2d, Conv2d) and isinstance(relu6, ReLU6), "conv2d has to be of Conv2d type and relu6 has to ReLU6 type"
    fused_layer = Conv2dReLU6(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = conv2d.bias is not None
    )
    fused_layer.weight.copy_(conv2d.weight)
    if conv2d.bias is not None and fused_layer.bias is not None:
        fused_layer.bias.copy_(conv2d.bias)
    return fused_layer



