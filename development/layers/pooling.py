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
    UINT8_T,
    UINT16_T,
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


def _pool_convert_to_c(layer, var_name, input_shape, for_arduino=False):
    """Shared C-code generator for MaxPool2d and AvgPool2d."""
    input_channel_size, input_row_size, input_col_size = input_shape
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding

    scheme = None
    if layer.is_quantized and "quantize" in layer.__dict__["_dmc"]:
        scheme = layer.__dict__["_dmc"]["quantize"]["scheme"]

    if scheme != QuantizationScheme.STATIC:
        params_info = [
            (UINT16_T, "input_channel", str(input_channel_size)),
            (UINT16_T, "input_row",     str(input_row_size)),
            (UINT16_T, "input_col",     str(input_col_size)),
            (UINT8_T,  "kernel_size",   str(kernel_size)),
            (UINT8_T,  "stride",        str(stride)),
        ]
        layer_def = layer.get_struct_def(var_name, params_info, QuantizationScheme.NONE, for_arduino)
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

        params_info = [
            (UINT16_T, "input_channel",    str(input_channel_size)),
            (UINT16_T, "input_row",        str(input_row_size)),
            (UINT16_T, "input_col",        str(input_col_size)),
            (UINT8_T,  "kernel_size",      str(kernel_size)),
            (UINT8_T,  "stride",           str(stride)),
            (UINT8_T,  "quantize_property", quantize_property),
        ]
        layer_def = layer.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
        layer_header = f"extern {layer.__class__.__name__}_SQ {var_name};\n\n"

    return layer_header, layer_def, ""


def _pool_workspace_locals(layer, scheme, ptr_size):
    """Returns (locals_size, runtime_size) for a pooling layer's forward() locals."""
    if scheme == QuantizationScheme.STATIC:
        # buffer scalars: 3×u16 + kernel u8 + stride u8 + quantize_property u8
        locals_size  = 9
        # computed: output_row u16 + output_col u16 = 4
        # loops: n u16 + m u16 + l u16 + j u8 + i u8 = 8
        # 2 fn ptrs + accumulator depends on pool type
        runtime_size = 14 + 2 * ptr_size  # max pool: temp i8 + input_val i8 = 2
    else:
        # buffer scalars: 3×u16 + kernel u8 + stride u8
        locals_size  = 8
        # computed: output_row u16 + output_col u16 = 4
        # loops: n u16 + m u16 + l u16 + j u8 + i u8 = 8
        # accumulator: temp f32 + input_val f32 = 8  (max pool)
        runtime_size = 20
    return locals_size, runtime_size


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

    def get_workspace_size(self, input_shape, data_per_byte,
                           include_locals=False, include_runtime=False, ptr_size=2) -> int:
        base = math.ceil(input_shape.numel() / data_per_byte)
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized and "quantize" in self.__dict__["_dmc"]:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
        locals_size, runtime_size = _pool_workspace_locals(self, scheme, ptr_size)
        return base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0)

    def get_output_tensor_shape(self, input_shape):
        return _pool_output_shape(input_shape, self.kernel_size, self.stride, self.padding)

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        return _pool_convert_to_c(self, var_name, input_shape, for_arduino=for_arduino)


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

    def get_workspace_size(self, input_shape, data_per_byte,
                           include_locals=False, include_runtime=False, ptr_size=2) -> int:
        base = math.ceil(input_shape.numel() / data_per_byte)
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized and "quantize" in self.__dict__["_dmc"]:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
        # AvgPool differs from MaxPool only in its accumulator: float total vs int16_t total
        # For STATIC: runtime = 15+2P (vs MaxPool's 14+2P: int16_t total=2 instead of 2×i8=2, same)
        # For float:  runtime = 17 (computed 5 with pool_size u8, loops 8, accumulator f32 4)
        if scheme == QuantizationScheme.STATIC:
            locals_size  = 9
            runtime_size = 15 + 2 * ptr_size  # pool_size u8 computed + int16_t total
        else:
            locals_size  = 8
            runtime_size = 17                  # pool_size u8 computed + float total
        return base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0)

    def get_output_tensor_shape(self, input_shape):
        return _pool_output_shape(input_shape, self.kernel_size, self.stride, self.padding)

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        return _pool_convert_to_c(self, var_name, input_shape, for_arduino=for_arduino)
