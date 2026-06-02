"""
@file pooling.py
@brief Pooling Layers (MaxPool2d, AvgPool2d) for DMC Pipeline.

In the Deep Microcompression framework, pooling layers are passive pass-throughs:
1. Structure Preservation: propagate pruning indices unchanged.
2. Quantization Pass-Through: MaxPool preserves quantization scales; AvgPool
   relies on float fallback or the C-library's handling.
"""

import math
from typing import Union, Optional

import torch
from torch import nn

from .layer import Layer
from ..compressors import Quantize, QuantizationScheme, QuantizationBitWidthError
from ..utils import (
    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2,
)


def _pool_output_shape(input_shape, kernel_size, stride, padding):
    """Shared spatial output-shape calculation for MaxPool2d and AvgPool2d."""
    C, H_in, W_in = input_shape

    def _pair(x):
        return x if isinstance(x, tuple) else (x, x)

    kH, kW = _pair(kernel_size)
    sH, sW = _pair(stride or kernel_size)  # PyTorch uses kernel_size as default stride
    pH, pW = _pair(padding)

    H_out = ((H_in + 2 * pH - kH) // sH) + 1
    W_out = ((W_in + 2 * pW - kW) // sW) + 1
    return torch.Size((C, H_out, W_out))


def _pool_convert_to_c(layer, var_name, input_shape):
    """Shared C-code generator for MaxPool2d and AvgPool2d."""
    input_channel_size, input_row_size, input_col_size = input_shape
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding

    scheme = None
    if layer.is_quantized and "quantize" in layer.__dict__["_dmc"]:
        scheme = layer.__dict__["_dmc"]["quantize"]["scheme"]

    if scheme != QuantizationScheme.STATIC:
        layer_def = (
            f"{layer.__class__.__name__} {var_name}("
            f"{input_channel_size}, {input_row_size}, {input_col_size}, "
            f"{kernel_size}, {stride}, {padding});\n"
        )
        layer_header = f"extern {layer.__class__.__name__} {var_name};\n\n"
    else:
        activation_bitwidth = layer.__dict__["_dmc"]["quantize"]["activation_bitwidth"]
        if activation_bitwidth == 8:
            quantize_property = ACTIVATION_BITWIDTH_8
        elif activation_bitwidth == 4:
            quantize_property = ACTIVATION_BITWIDTH_4
        elif activation_bitwidth == 2:
            quantize_property = ACTIVATION_BITWIDTH_2
        else:
            raise QuantizationBitWidthError(activation_bitwidth)

        layer_def = (
            f"{layer.__class__.__name__}_SQ {var_name}("
            f"{input_channel_size}, {input_row_size}, {input_col_size}, "
            f"{kernel_size}, {stride}, {padding}, {quantize_property});\n"
        )
        layer_header = f"extern {layer.__class__.__name__}_SQ {var_name};\n\n"

    return layer_header, layer_def, ""


class MaxPool2d(Layer, nn.MaxPool2d):
    """
    DMC-aware MaxPool2d layer.

    Forwards pruning indices and quantization scale unchanged — max over integers
    is identical to max over floats at the same scale, so no re-quantization needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    def get_size_in_bits(self):
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return (math.ceil(input_shape.numel() / data_per_byte)
                + math.ceil(self.get_output_tensor_shape(input_shape).numel() / data_per_byte))

    def get_output_tensor_shape(self, input_shape):
        return _pool_output_shape(input_shape, self.kernel_size, self.stride, self.padding)

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        return _pool_convert_to_c(self, var_name, input_shape)


class AvgPool2d(Layer, nn.AvgPool2d):
    """
    DMC-aware AvgPool2d layer.

    Note on Quantization: Average Pooling introduces non-integer values (sum/count).
    In a strict integer-only pipeline, this requires rescaling. The current
    implementation relies on the C-library's handling or assumes float fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    def get_size_in_bits(self):
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return (math.ceil(input_shape.numel() / data_per_byte)
                + math.ceil(self.get_output_tensor_shape(input_shape).numel() / data_per_byte))

    def get_output_tensor_shape(self, input_shape):
        return _pool_output_shape(input_shape, self.kernel_size, self.stride, self.padding)

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        return _pool_convert_to_c(self, var_name, input_shape)
