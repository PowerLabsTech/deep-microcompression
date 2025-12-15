"""
@file activation.py
@brief PyTorch implementation of ReLU layer with support for:
    1. Standard floating-point operation
    2. Static quantization (per-tensor and per-channel)
    3. C code generation for deployment
"""

from typing import Optional

import torch
from torch import nn

from ..utils import (
    quantize_per_tensor_assy,
    get_size_in_bits,

    convert_tensor_to_bytes_var
)
from .layer import Layer
from ..compressors import (
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
)


class ReLU(Layer, nn.ReLU):
    """Quantization-aware ReLU layer with support for:
        - Standard ReLU operation
        - Quantized inference modes
        - Model pruning (placeholder)
        - C code generation for deployment
    """

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
    

    def init_quantize(self, bitwidth, scheme, granularity, previous_output_quantize = None):

        if scheme == QuantizationScheme.STATIC:
            # raise RuntimeError("Can not perform static quantization with ReLU, fuse the model first!")
            setattr(self, "input_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC, base=[previous_output_quantize]
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
        return input_shape, input_shape
    
    
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
        input_size = input_shape.numel()

        layer_param_def = ""
        layer_header = ""

        scheme = None
        if self.is_quantized and hasattr(self, "input_quantize"):
            scheme = self.input_quantize.scheme

        if scheme != QuantizationScheme.STATIC:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
        else:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size}, *(int8_t*){var_name}_input_zero_point);\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, 
                f"{var_name}_input_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def
        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def
    


class ReLU6(Layer, nn.ReLU6):
    """
    Quantization-aware ReLU6 layer.
    
    Common in mobile-optimized networks (e.g., MobileNet).
    
    Quantization Logic:
    Standard ReLU6: y = min(max(0, x), 6)
    Quantized ReLU6: y_q = min(max(zero_point, x_q), six_point)
    
    Requires calculating 'six_point': the integer representation of 6.0
    given the current scale and zero_point.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass with quantization awareness"""
        return super().forward(input)
    

    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        # Propagate channel indices (Structured Pruning Support)
        return keep_prev_channel_index


    def get_prune_channel_possible_hyperparameters(self):
        return None
    
    def init_quantize(self, bitwidth, scheme, granularity, previous_output_quantize = None):
        """Configures quantization for integer clamping."""
        if scheme == QuantizationScheme.STATIC:
            # raise RuntimeError("Can not perform static quantization with ReLU6, fuse the model first!")
            setattr(self, "input_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC, base=[previous_output_quantize]
            ))
            return previous_output_quantize

    def get_size_in_bits(self):
        if self.is_quantized:
            if hasattr(self, "input_quantize"):
                return get_size_in_bits(self.input_quantize.zero_point)*2
        return 0

    def get_compression_parameters(self):
        # Nothing to do 
        pass


    def get_output_tensor_shape(self, input_shape):
        # Nothing to do
        return input_shape, input_shape
    
    
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
        input_size = input_shape.numel()


        layer_param_def = ""
        layer_header = ""

        scheme = None
        if self.is_quantized and hasattr(self, "input_quantize"):
            scheme = self.input_quantize.scheme

        if scheme != QuantizationScheme.STATIC:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
        else:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size}, *(int8_t*){var_name}_input_zero_point, *(int8_t*){var_name}_input_six_point);\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, 
                f"{var_name}_input_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def

            input_six_point = quantize_per_tensor_assy(torch.Tensor([6]), self.input_quantize.scale, self.input_quantize.zero_point, self.input_quantize.bitwidth)
            param_header, param_def = convert_tensor_to_bytes_var(
                input_six_point.to(torch.int8), 
                f"{var_name}_input_six_point"
            )
            layer_header += param_header
            layer_param_def += param_def

        # layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def
    