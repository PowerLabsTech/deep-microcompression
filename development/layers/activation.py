import math
import warnings
from typing import Optional

import torch
from torch import nn

from ..utils import (
    quantize_per_tensor_assy,
    get_size_in_bits,
    get_data_bits,
    pad_bits_to_byte,

    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2,

    UINT8_T,
    UINT32_T,
    INT8_T
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

    def get_workspace_size(
        self, input_shape, include_locals=False,
        include_runtime=False, ptr_size=2
    ) -> int:
        data_bits = get_data_bits(self)
        base = pad_bits_to_byte(input_shape.numel() * data_bits)
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized:
            scheme = self.input_quantize.scheme
        if scheme == QuantizationScheme.STATIC:
            locals_size  = 6               # uint32_t input_size + int8_t zero_point + uint8_t property
            runtime_size = 4 + 2 * ptr_size  # uint32_t i + 2 fn ptrs
        else:
            locals_size  = 4               # uint32_t input_size
            runtime_size = 4               # uint32_t i
        return base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0)


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
            params_info = [
                (UINT32_T, "input_size", str(input_size)),
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

            zero_point_val = int(self.input_quantize.zero_point.item())
            params_info = [
                (UINT32_T, "input_size",        str(input_size)),
                (INT8_T,   "input_zero_point",  str(zero_point_val)),
                (UINT8_T,  "quantize_property", quantize_property),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

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

    def get_workspace_size(
        self, input_shape, include_locals=False,
        include_runtime=False, ptr_size=2
    ) -> int:
        data_bits = get_data_bits(self)
        base = pad_bits_to_byte(input_shape.numel() * data_bits)
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized:
            scheme = self.input_quantize.scheme
        if scheme == QuantizationScheme.STATIC:
            locals_size  = 7               # uint32_t input_size + int8_t zero_point + int8_t six_point + uint8_t property
            runtime_size = 4 + 2 * ptr_size  # uint32_t i + 2 fn ptrs
        else:
            locals_size  = 4               # uint32_t input_size
            runtime_size = 4               # uint32_t i
        return base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0)

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
            params_info = [
                (UINT32_T, "input_size", str(input_size)),
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

            zero_point_val = int(self.input_quantize.zero_point.item())
            input_six_point = quantize_per_tensor_assy(
                torch.Tensor([6]).to(device=self.input_quantize.scale.device),
                self.input_quantize.scale, self.input_quantize.zero_point,
                self.input_quantize.bitwidth,
            )
            six_point_val = int(input_six_point.item())
            params_info = [
                (UINT32_T, "input_size",        str(input_size)),
                (INT8_T,   "input_zero_point",  str(zero_point_val)),
                (INT8_T,   "input_six_point",   str(six_point_val)),
                (UINT8_T,  "quantize_property", quantize_property),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

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

    def get_workspace_size(self, input_shape) -> int:
        data_bits = get_data_bits(self)
        return pad_bits_to_byte(input_shape.numel() * data_bits)

    def get_output_tensor_shape(self, input_shape):
        return input_shape

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        raise NotImplementedError(
            "Dropout.convert_to_c is not implemented: Dropout is a training-only layer "
            "and must be removed before C export. Call model.fuse() or remove Dropout layers."
        )
