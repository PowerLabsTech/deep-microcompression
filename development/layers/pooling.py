"""
@file pooling.py
@brief PyTorch implementation of MaxPool2d layer with support for:
    1. Standard max pooling operation
    2. Static quantization (per-tensor and per-channel)
    3. C code generation for deployment
"""
"""
@file pooling.py
@brief Pooling Layers (MaxPool2d, AvgPool2d) for DMC Pipeline.

In the Deep Microcompression framework, pooling layers play a passive but critical role:
1.  Structure Preservation: They must propagate pruning indices.
    If Channel $k$ is pruned in `Conv_1`, then Channel $k$ in `MaxPool_1` is also dead,
    and this information must be passed to `Conv_2`.
2.  Quantization Pass-Through: MaxPool preserves quantization
    scales (max of integers is effectively the same scale). AvgPool requires
    re-quantization support (not implemented here, assumed fused or float fallback).
"""

from typing import Union

import torch
from torch import nn

from .layer import Layer

class MaxPool2d(Layer, nn.MaxPool2d):
    """
    DMC-aware MaxPool2d layer.
    
    Responsibilities:
    1.  Pruning Coordination: Forwards 'keep_channel_mask' unchanged.
    2.  Quantization: Inherits scale/zero-point from previous layer (Max operation
        on quantized int8 values works identical to float values).
    3.  C-Generation: Exports kernel/stride/padding parameters for the bare-metal loop.
    """

    def __init__(self, *args, **kwargs):
        """Initialize MaxPool2d layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """
        Note: Since MaxPool operates on individual values, it natively supports 
        Quantized Integers without modification (Max(int8) is valid).
        """
        return super().forward(input)

    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Union[torch.Tensor, None], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """Placeholder for channel pruning (MaxPool doesn't have weights to prune)"""
        return keep_prev_channel_index


    def get_prune_channel_possible_hyperparameters(self):
        return None

    def init_quantize(self, bitwidth, scheme, granularity, previous_output_quantize = None):
        """
        Pass-through for Quantization Observers.
        
        Max Pooling does not alter the dynamic range of data, so the 
        input scale/zero-point is valid for the output.
        """
        return previous_output_quantize

    def get_size_in_bits(self):
        return 0


    def get_compression_parameters(self):
        #Nothing to do
        pass


    def get_output_tensor_shape(self, input_shape):
        """
        Calculates output spatial dimensions.
        Used for SRAM workspace estimation (Section IV-A-3).
        """
        C, H_in, W_in = input_shape
        
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        kH, kW = _pair(self.kernel_size)
        sH, sW = _pair(self.stride or self.kernel_size)  # PyTorch uses kernel_size as default if stride is None
        pH, pW = _pair(self.padding)
        
        # print(H_in, pH, kW, sW, self.kernel_size, self.stride, self.padding, self.dilation, isinstance(self.kernel_size, tuple))
        
        H_out = ((H_in + 2 * pH - kH) // sH) + 1
        W_out = ((W_in + 2 * pW - kW) // sW) + 1
        
        return torch.Size((C, H_out, W_out)), torch.Size((C, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """
        Generates C code for bare-metal deployment.

        Args:
            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if needed
        
        Exports structural parameters (Kernel, Stride, Padding) so the 
        generic C implementation can execute the loop.
        """
        input_channel_size, input_row_size, input_col_size = input_shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        layer_def = (
            f"{self.__class__.__name__} {var_name}("
            f"{input_channel_size}, {input_row_size}, {input_col_size}, "
            f"{kernel_size}, {stride}, {padding});\n"
        )
        layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
        layer_param_def = ""

        return layer_header, layer_def, layer_param_def
    



class AvgPool2d(Layer, nn.AvgPool2d):
    """
    DMC-aware AvgPool2d layer.
    
    Note on Quantization: Average Pooling introduces non-integer values (sums/count).
    In a strict integer-only pipeline (Static Quantization), this requires 
    rescaling. The current implementation relies on the C-library's handling 
    or assumes float fallback for AvgPool layers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass through max pooling layer
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Max pooled output tensor
        """
        return super().forward(input)

    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Union[torch.Tensor, None], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """Placeholder for channel pruning (MaxPool doesn't have weights to prune)"""
        # Nothing to do
        return keep_prev_channel_index


    def get_prune_channel_possible_hyperparameters(self):
        return None
    

    def init_quantize(self, bitwidth, scheme, granularity, previous_output_quantize = None):
        """Pass-through for quantization observers."""
        return previous_output_quantize


    def get_size_in_bits(self):
        return 0


    def get_compression_parameters(self):
        #Nothing to do
        pass


    def get_output_tensor_shape(self, input_shape):
        
        C, H_in, W_in = input_shape
        
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        kH, kW = _pair(self.kernel_size)
        sH, sW = _pair(self.stride or self.kernel_size)  # PyTorch uses kernel_size as default if stride is None
        pH, pW = _pair(self.padding)
        
        # print(H_in, pH, kW, sW, self.kernel_size, self.stride, self.padding, self.dilation, isinstance(self.kernel_size, tuple))
        
        H_out = ((H_in + 2 * pH - kH) // sH) + 1
        W_out = ((W_in + 2 * pW - kW) // sW) + 1

        return torch.Size((C, H_out, W_out)), torch.Size((C, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """Generate C code declarations for this layer
        
        Args:
            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if needed

        Returns:
            Tuple of (header declaration, layer definition, parameter definition)
        """
        input_channel_size, input_row_size, input_col_size = input_shape
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        layer_def = (
            f"{self.__class__.__name__} {var_name}("
            f"{input_channel_size}, {input_row_size}, {input_col_size}, "
            f"{kernel_size}, {stride}, {padding});\n"
        )
        layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
        layer_param_def = ""

        return layer_header, layer_def, layer_param_def