import math
import warnings
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


def _attach_input_quantize(layer, activation_bitwidth, scheme, previous_output_quantize, current_output_quantize):
    """Shared helper: pick the calibration base and attach input_quantize."""
    quantization_base = current_output_quantize if current_output_quantize is not None else previous_output_quantize
    setattr(layer, "input_quantize", Quantize(
        layer, activation_bitwidth, scheme,
        QuantizationGranularity.PER_TENSOR,
        scale_type=QuantizationScaleType.ASSYMMETRIC,
        base=[quantization_base],
    ))
    return quantization_base


class ReLU(Layer, nn.ReLU):
    """Quantization-aware ReLU layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    def init_quantize(self, parameter_bitwidth, granularity, scheme,
                      activation_bitwidth=None, previous_output_quantize=None,
                      current_output_quantize: Optional[Quantize] = None):
        super().init_quantize(parameter_bitwidth, granularity, scheme,
                              activation_bitwidth, previous_output_quantize)
        if scheme == QuantizationScheme.STATIC:
            assert activation_bitwidth is not None, \
                "Pass an activation bitwidth when doing static quantization"
            return _attach_input_quantize(
                self, activation_bitwidth, scheme,
                previous_output_quantize, current_output_quantize,
            )

    def get_size_in_bits(self):
        if self.is_quantized:
            return get_size_in_bits(self.input_quantize.zero_point)
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return math.ceil(input_shape.numel() / data_per_byte)

    def get_output_tensor_shape(self, input_shape):
        return input_shape

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        input_size = input_shape.numel()
        layer_param_def = ""
        layer_header = ""

        scheme = None
        if self.is_quantized:
            scheme = self.input_quantize.scheme

        if scheme != QuantizationScheme.STATIC:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
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

            layer_def = (f"{self.__class__.__name__}_SQ {var_name}"
                         f"({input_size}, *(int8_t*){var_name}_input_zero_point, {quantize_property});\n")
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, f"{var_name}_input_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def

        return layer_header, layer_def, layer_param_def


class ReLU6(Layer, nn.ReLU6):
    """
    Quantization-aware ReLU6 layer.

    Quantized ReLU6: y_q = min(max(zero_point, x_q), six_point)
    where six_point is the integer encoding of 6.0 under the current scale/zero_point.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    def init_quantize(self, parameter_bitwidth, granularity, scheme,
                      activation_bitwidth=None, previous_output_quantize=None,
                      current_output_quantize: Optional[Quantize] = None):
        super().init_quantize(parameter_bitwidth, granularity, scheme,
                              activation_bitwidth, previous_output_quantize)
        if scheme == QuantizationScheme.STATIC:
            assert activation_bitwidth is not None, \
                "Pass an activation bitwidth when doing static quantization"
            return _attach_input_quantize(
                self, activation_bitwidth, scheme,
                previous_output_quantize, current_output_quantize,
            )

    def get_size_in_bits(self):
        if self.is_quantized:
            # zero_point (int8) + six_point (int8)
            return (get_size_in_bits(self.input_quantize.zero_point)
                    + get_size_in_bits(self.input_quantize.zero_point))
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return math.ceil(input_shape.numel() / data_per_byte)

    def get_output_tensor_shape(self, input_shape):
        return input_shape

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        input_size = input_shape.numel()
        layer_param_def = ""
        layer_header = ""

        scheme = None
        if self.is_quantized:
            scheme = self.input_quantize.scheme

        if scheme != QuantizationScheme.STATIC:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
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

            layer_def = (f"{self.__class__.__name__}_SQ {var_name}"
                         f"({input_size}, *(int8_t*){var_name}_input_zero_point, "
                         f"*(int8_t*){var_name}_input_six_point, {quantize_property});\n")
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, f"{var_name}_input_zero_point")
            layer_header += param_header
            layer_param_def += param_def

            input_six_point = quantize_per_tensor_assy(
                torch.Tensor([6]).to(device=self.input_quantize.scale.device),
                self.input_quantize.scale, self.input_quantize.zero_point,
                self.input_quantize.bitwidth,
            )
            param_header, param_def = convert_tensor_to_bytes_var(
                input_six_point.to(torch.int8), f"{var_name}_input_six_point")
            layer_header += param_header
            layer_param_def += param_def

        return layer_header, layer_def, layer_param_def


class Dropout(Layer, nn.Dropout):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    def init_quantize(self, parameter_bitwidth, granularity, scheme,
                      activation_bitwidth=None, previous_output_quantize=None,
                      current_output_quantize: Optional[Quantize] = None):
        super().init_quantize(parameter_bitwidth, granularity, scheme,
                              activation_bitwidth, previous_output_quantize)
        if scheme == QuantizationScheme.STATIC:
            assert activation_bitwidth is not None, \
                "Pass an activation bitwidth when doing static quantization"
            return _attach_input_quantize(
                self, activation_bitwidth, scheme,
                previous_output_quantize, current_output_quantize,
            )

    def get_size_in_bits(self):
        if self.is_quantized:
            return get_size_in_bits(self.input_quantize.zero_point)
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return math.ceil(input_shape.numel() / data_per_byte)

    def get_output_tensor_shape(self, input_shape):
        return input_shape

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        raise NotImplementedError(
            "Dropout.convert_to_c is not implemented: Dropout is a training-only layer "
            "and must be removed before C export. Call model.fuse() or remove Dropout layers."
        )
