from typing import Optional

import torch
from torch import nn

from ..utils import (
    quantize_per_tensor_assy,
    get_size_in_bits,
    convert_tensor_to_bytes_var,

    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2
)

from .layer import Layer
from ..compressors import (
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
)


class ConstantPad2d(Layer, nn.ConstantPad2d):

    def __init__(self, *args, **kwargs):
        """Initialize ReLU layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Clamped output tensor according to current quantization mode
        """

        return super().forward(input)
    
    @torch.no_grad()
    def init_prune_channel(self, 
                     sparsity: float, 
                     keep_prev_channel_index: Optional[torch.Tensor], 
                     input_shape: torch.Size,
                     is_output_layer: bool = False, 
                     metric: str = "l2"):
        """Placeholder for channel pruning functionality
        
        Args:
            sparsity: Target sparsity ratio (unused)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Pruning metric (unused)
            
        Returns:
            Original channel indices (no pruning implemented)
        """
        # Nothing to do
        return keep_prev_channel_index
    

    def get_prune_channel_possible_hyperparameters(self):
        return None


    def get_quantize_possible_hyperparameters(self):
        return None


    def init_quantize(self, parameter_bitwidth, granularity, scheme, activation_bitwidth=None, previous_output_quantize = None):
        super().init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)

        if scheme == QuantizationScheme.STATIC:
            # raise RuntimeError("Can not perform static quantization with ReLU, fuse the model first!")
            assert activation_bitwidth is not None, "Pass an activation bitwidth when doing static quantization"
            setattr(self, "input_quantize", Quantize(
                self, activation_bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC, base=[previous_output_quantize]
            ))
            return previous_output_quantize


    def get_size_in_bits(self):
        if self.is_quantized:
            if hasattr(self, "input_quantize"):
                return get_size_in_bits(self.input_quantize.zero_point)
        return 0
    

    def get_compression_parameters(self):
        # Nothing to do 
        pass


    def get_output_tensor_shape(self, input_shape):
        # Nothing to do
        C_in, H_in, W_in = input_shape
        pW = self.padding[0] + self.padding[1]
        pH = self.padding[2] + self.padding[3]
        return input_shape, torch.Size((C_in, H_in +  pH, W_in +  pW))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """Generates C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if needed
            
        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_channel_size, input_row_size, input_col_size = input_shape

        layer_param_def = ""
        layer_header = ""
        padding = self.padding

        scheme = None
        if self.is_quantized and hasattr(self, "input_quantize"):
            scheme = self.input_quantize.scheme

        if scheme != QuantizationScheme.STATIC:
            layer_def = (
                f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                f"{input_row_size}, {input_col_size}, {self.value}, "
                "{" f"{padding[0]}, {padding[1]}, {padding[2]}, {padding[3]}" "});\n" 
            )
            layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        else:

            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
            activation_bitwidth = self.__dict__["_dmc"]["quantize"]["activation_bitwidth"]

            quantize_property = ""

            if activation_bitwidth == 8:
                quantize_property += ACTIVATION_BITWIDTH_8
            elif activation_bitwidth == 4:
                quantize_property += ACTIVATION_BITWIDTH_4
            elif activation_bitwidth == 2:
                quantize_property += ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError
            
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
                input_value_point.to(torch.int8), 
                f"{var_name}_input_value_point"
            )

            layer_header += param_header
            layer_param_def += param_def

        return layer_header, layer_def, layer_param_def
    
