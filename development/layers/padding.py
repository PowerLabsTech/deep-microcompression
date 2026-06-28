import math
from typing import Optional

import torch
from torch import nn

from ..utils import (
    quantize_per_tensor_assy,
    get_size_in_bits,
    convert_tensor_to_bytes_var,
    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2,
    UINT8_T,
    UINT16_T,
    INT8_T,
    FLOAT_T,
)

from .layer import Layer
from ..compressors import (
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
    QuantizationBitWidthError,
)


class ConstantPad2d(Layer, nn.ConstantPad2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    def init_quantize(
        self,
        parameter_bitwidth,
        granularity,
        scheme,
        activation_bitwidth=None,
        previous_output_quantize=None,
        current_output_quantize: Optional[Quantize] = None,
    ):
        # For STATIC: attach an input_quantize so we can encode the pad value as int8.
        # For all other schemes: use the base class passthrough.
        if scheme != QuantizationScheme.STATIC:
            return super().init_quantize(
                parameter_bitwidth, granularity, scheme,
                activation_bitwidth, previous_output_quantize, current_output_quantize,
            )
        super().init_quantize(parameter_bitwidth, granularity, scheme,
                              activation_bitwidth, previous_output_quantize)
        assert activation_bitwidth is not None, \
            "Pass an activation bitwidth when doing static quantization"
        quantization_base = (current_output_quantize
                             if current_output_quantize is not None
                             else previous_output_quantize)
        setattr(self, "input_quantize", Quantize(
            self, activation_bitwidth, scheme,
            QuantizationGranularity.PER_TENSOR,
            scale_type=QuantizationScaleType.ASSYMMETRIC,
            base=[quantization_base],
        ))
        return quantization_base

    def get_size_in_bits(self):
        if self.is_quantized:
            return get_size_in_bits(self.input_quantize.zero_point)
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte,
                           include_locals=False, include_runtime=False, ptr_size=2) -> int:
        base = math.ceil(self.get_output_tensor_shape(input_shape).numel() / data_per_byte)
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized:
            scheme = self.input_quantize.scheme
        if scheme == QuantizationScheme.STATIC:
            # 3×u16 + i8 value_point + Padding_t(4×u8) + u8 property
            locals_size  = 12
            # computed: padded_row u16 + padded_col u16 = 4; loops: 3×i32 = 12; fn ptrs: 2P
            runtime_size = 16 + 2 * ptr_size
        else:
            # 3×u16 + f32 value + Padding_t(4×u8)
            locals_size  = 14
            # computed: padded_row u16 + padded_col u16 = 4; loops: 3×i32 = 12
            runtime_size = 16
        return base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0)

    def get_output_tensor_shape(self, input_shape):
        # padding order: (left, right, top, bottom)
        C_in, H_in, W_in = input_shape
        pW = self.padding[0] + self.padding[1]
        pH = self.padding[2] + self.padding[3]
        return torch.Size((C_in, H_in + pH, W_in + pW))

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        input_channel_size, input_row_size, input_col_size = input_shape
        layer_param_def = ""
        layer_header = ""
        padding = self.padding

        scheme = None
        if self.is_quantized:
            scheme = self.input_quantize.scheme

        if scheme != QuantizationScheme.STATIC:
            params_info = [
                (UINT16_T, "input_channel_size", str(input_channel_size)),
                (UINT16_T, "input_row_size",     str(input_row_size)),
                (UINT16_T, "input_col_size",     str(input_col_size)),
                (UINT8_T,  "_pad0",              "0"),
                (UINT8_T,  "_pad1",              "0"),
                (FLOAT_T,  "value",              f"{float(self.value):.9g}f"),
                (UINT8_T,  "padding_left",       str(padding[0])),
                (UINT8_T,  "padding_right",      str(padding[1])),
                (UINT8_T,  "padding_top",        str(padding[2])),
                (UINT8_T,  "padding_bottom",     str(padding[3])),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.NONE, for_arduino)
            layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        else:
            activation_bitwidth = self.__dict__["_dmc"]["quantize"]["activation_bitwidth"]
            if activation_bitwidth == 8:
                quantize_property = ACTIVATION_BITWIDTH_8
            elif activation_bitwidth == 4:
                quantize_property = ACTIVATION_BITWIDTH_4
            elif activation_bitwidth == 2:
                quantize_property = ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(activation_bitwidth)

            input_value_point = quantize_per_tensor_assy(
                torch.Tensor([self.value]).to(device=self.input_quantize.scale.device),
                self.input_quantize.scale, self.input_quantize.zero_point,
                self.input_quantize.bitwidth
            )
            input_value_point_val = int(input_value_point.item())
            params_info = [
                (UINT16_T, "input_channel_size", str(input_channel_size)),
                (UINT16_T, "input_row_size",     str(input_row_size)),
                (UINT16_T, "input_col_size",     str(input_col_size)),
                (INT8_T,   "input_value_point",  str(input_value_point_val)),
                (UINT8_T,  "padding_left",       str(padding[0])),
                (UINT8_T,  "padding_right",      str(padding[1])),
                (UINT8_T,  "padding_top",        str(padding[2])),
                (UINT8_T,  "padding_bottom",     str(padding[3])),
                (UINT8_T,  "quantize_property",  quantize_property),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

        return layer_header, layer_def, layer_param_def
