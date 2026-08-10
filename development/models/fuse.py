"""
@file fuse.py
@brief Layer Fusion Utilities for Bare-Metal Optimization

Merges adjacent layer pairs into unified fused layers and transfers DMC
pipeline state (pruning masks, quantization observers) to the new objects.

Supported fusions:
  Conv2d + BatchNorm2d  -> Conv2d       (BN folded into weights/bias)
  Conv2d + ReLU         -> Conv2dReLU
  Conv2d + ReLU6        -> Conv2dReLU6
  Linear + ReLU         -> LinearReLU
  Linear + ReLU6        -> LinearReLU6

Role in DMC Pipeline:
1.  Pre-Quantization: BN folding is required before QAT so that integer
    scales are calibrated against the true effective weight distribution,
    not the pre-fold weights.
2.  Bias synthesis: When Conv2d had bias=False and BN folding introduces
    a bias, fuse_conv2d_batchnorm2d synthesises bias_prune_channel (from
    BN's existing prune_channel) and bias_quantize (scale = s_w * s_in)
    so the fused layer's invariants match a layer that had bias=True all
    along.
3.  Inference optimisation: Fewer layer objects reduce function-call
    overhead and intermediate SRAM buffers in the generated C library.
"""
import torch

from ..layers.conv import Conv2d
from ..layers.batchnorm import BatchNorm2d
from ..layers.linear import Linear
from ..layers.activation import ReLU, ReLU6
from ..layers.fused_layers import LinearReLU, Conv2dReLU, LinearReLU6, Conv2dReLU6
from ..compressors import Quantize, QuantizationScheme, QuantizationScaleType
from ..utils import STATIC_BIAS_BITWDHT


@torch.no_grad()
def init_dmc_parameter(original_layer, fused_layer):
    """
    Copies all DMC pipeline state from original_layer to fused_layer.

    Covers pruning masks (weight/bias_prune_channel, is_pruned_channel),
    quantization observers (weight/bias/input/output_quantize, is_quantized),
    and the _dmc metadata dict. Only attributes that exist on original_layer
    are copied — missing attributes are left unset on fused_layer.
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
    Folds BatchNorm2d statistics into Conv2d weights and bias.

    Transforms:
        y = BN(W*x + b)  =  W'*x + b'
        W' = W * gamma / sqrt(var + eps)
        b' = (b - mean) * gamma / sqrt(var + eps) + beta

    When conv2d.bias is None the fused bias is entirely new. In that case
    this function also synthesises bias_prune_channel (reusing BN's existing
    prune_channel, which holds the same output-channel keep-indices) and
    bias_quantize (scale = s_w * s_in, identical to what init_quantize would
    have created had the bias existed at quantization time).

    Args:
        conv2d: Source Conv2d layer (already pruned/quantized if applicable).
        batchnorm2d: BatchNorm2d that immediately follows conv2d.

    Returns:
        A new Conv2d with bias=True containing the fused parameters.
    """
    assert isinstance(conv2d, Conv2d) and isinstance(batchnorm2d, BatchNorm2d), "conv2d has to be of Conv2d type and batchnorm2d has to be BatchNorm2d type"
    assert conv2d.out_channels == batchnorm2d.num_features, f"conv2d and batchnorm not fuseable, conv2d has {conv2d.out_channels} out_channels and batchnorm2d has {batchnorm2d.num_features} num_features, the must tbe equal"
    fused_layer = Conv2d(
        out_channels = conv2d.out_channels,
        in_channels = conv2d.in_channels,
        kernel_size = conv2d.kernel_size,
        stride = conv2d.stride,
        padding=conv2d.padding,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = True
    )
    fused_layer.weight.copy_(conv2d.weight * batchnorm2d.folded_weight.view(-1,1,1,1))
    if conv2d.bias is not None:
        fused_layer.bias.copy_(conv2d.bias * batchnorm2d.folded_weight + batchnorm2d.folded_bias) # type: ignore
    else:
        fused_layer.bias.copy_(batchnorm2d.folded_bias) # type: ignore
        # Bias is new (conv had bias=False). BN's prune_channel already holds the right
        # output-channel keep-indices — reuse it directly instead of creating a duplicate.
        if hasattr(batchnorm2d, "prune_channel"):
            fused_layer.bias_prune_channel = batchnorm2d.prune_channel

        if hasattr(conv2d, "weight_quantize") and hasattr(conv2d, "input_quantize"):
            dmc_q = conv2d.__dict__.get("_dmc", {}).get("quantize", {})
            scheme = dmc_q.get("scheme")
            granularity = dmc_q.get("granularity")
            if scheme == QuantizationScheme.STATIC and granularity is not None:
                fused_layer.bias_quantize = Quantize(
                    fused_layer, STATIC_BIAS_BITWDHT, scheme, granularity,
                    scale_type=QuantizationScaleType.SYMMETRIC,
                    base=[conv2d.weight_quantize, conv2d.input_quantize],
                    prune_channel=getattr(fused_layer, "bias_prune_channel", None),
                )
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
        padding=conv2d.padding,
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
        padding=conv2d.padding,
        groups = conv2d.groups,
        dilation = conv2d.dilation,
        bias = conv2d.bias is not None
    )
    fused_layer.weight.copy_(conv2d.weight)
    if conv2d.bias is not None and fused_layer.bias is not None:
        fused_layer.bias.copy_(conv2d.bias)
    return fused_layer



