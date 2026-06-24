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

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return math.ceil(self.get_output_tensor_shape(input_shape).numel() / data_per_byte)

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
            layer_def = (
                f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                f"{input_row_size}, {input_col_size}, {self.value}, "
                "{" f"{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}" "});\n"
            )
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

            layer_def = (
                f"{self.__class__.__name__}_SQ {var_name}({input_channel_size}, "
                f"{input_row_size}, {input_col_size}, *(int8_t*){var_name}_input_value_point, "
                "{" f"{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}" "}, "
                f"{quantize_property});\n"
            )
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

            input_value_point = quantize_per_tensor_assy(
                torch.Tensor([self.value]).to(device=self.input_quantize.scale.device),
                self.input_quantize.scale, self.input_quantize.zero_point,
                self.input_quantize.bitwidth
            )
            param_header, param_def = convert_tensor_to_bytes_var(
                input_value_point.to(torch.int8), f"{var_name}_input_value_point")
            layer_header += param_header
            layer_param_def += param_def

        return layer_header, layer_def, layer_param_def
