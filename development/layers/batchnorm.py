"""
@file batchnorm.py
@brief BatchNorm2d Layer with folding and pruning support.

In the Deep Microcompression (DMC) pipeline, BatchNorm layers are transient. 
For the final bare-metal deployment, they are typically folded into the 
preceding Conv2d layer (see `fuse.py`) to enable Static Quantization .
"""

from typing import Optional

import torch
from torch import nn

from .layer import Layer
from ..compressors import Prune_Channel

from ..utils import (
    convert_tensor_to_bytes_var,
    get_size_in_bits,
)
from ..compressors import QuantizationScheme

class BatchNorm2d(Layer, nn.BatchNorm2d):
    """
    Extended BatchNorm2d with support for:
    - Weight Folding (Pre-calculation of effective weights for fusion)
    - Channel Pruning alignment
    - C-Code generation (Floating-point fallback)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def forward(self, input):
        return super().forward(input)
    
    # integer requirement, BN is mathematically folded into Conv weights.
    # y = (x - mean) / sqrt(var + eps) * gamma + beta
    # Becomes: y = x * scale + bias_shift
    @property
    def folded_weight(self):
        """
        Calculates the effective weight scaling factor.
        Formula: gamma / sqrt(running_var + eps)
        """
        if self.running_var is not None:
            return self.weight / torch.sqrt(self.running_var + self.eps)
        return self.weight

    @property
    def folded_bias(self):
        """
        Calculates the effective bias shift.
        Formula: beta - (running_mean * gamma) / sqrt(running_var + eps)
        """
        if self.running_mean is not None and self.running_var is not None:
            return self.bias - self.running_mean * self.weight / torch.sqrt(self.running_var + self.eps)
        return self.bias
   
    @torch.no_grad
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """
         aligns BatchNorm statistics with the pruned channels of the previous layer.

        In Structured Pruning, if a filter is removed from the 
        preceding Conv2d, the corresponding channel in BatchNorm is 'dead'.
        This method registers a Prune_Channel mask to ensure those dead statistics 
        are not exported.
        """
        # BatchNorm doesn't reduce channels itself; it just mirrors the input.
        # So it passes the 'keep' indices forward unchanged.
        setattr(self, "prune_channel", Prune_Channel(
            layer=self, keep_current_channel_index=keep_prev_channel_index
        ))
        return keep_prev_channel_index
    

    def get_prune_channel_possible_hyperparameters(self):
        return None


    @torch.no_grad()
    def init_quantize(self, parameter_bitwidth, granularity, scheme, activation_bitwidth=None, previous_output_quantize = None):
        """
        Configures quantization.
        
        NOTE: In the optimized DMC pipeline (Static Quantization), this method 
        should ideally not be reached because `fuse.py` should have merged this 
        layer into a Conv2d/Linear layer. 
        
        If this runs, it implies the user is running a floating-point baseline 
        or dynamic quantization.
        """
        super().init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)
        # if scheme == QuantizationScheme.STATIC:
        #     raise RuntimeError("Can not perform static quantization with BatchNorm2d, fuse the model first!")
            
        return previous_output_quantize


    def get_quantize_possible_hyperparameters(self):
        return None
    

    def get_size_in_bits(self):
        """Calculates size of the effective (folded) parameters."""
        folded_weight, folded_bias = self.get_compression_parameters()
        size = 0

        size += get_size_in_bits(folded_weight)
        size += get_size_in_bits(folded_bias)

        return size

    def get_compression_parameters(self):
        """Retrieves parameters, applying pruning masks if active."""
        folded_weight = self.folded_weight
        folded_bias = self.folded_bias

        if self.is_compressed:

            if self.is_pruned_channel:
                folded_weight = self.prune_channel.apply(folded_weight)
                folded_bias = self.prune_channel.apply(folded_bias)

        return folded_weight, folded_bias

    def get_output_tensor_shape(self, input_shape):
        # Nothing to do
        return input_shape, input_shape

    
    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """
        Generates C code for a standalone BatchNorm layer.

        Args:
            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if

        WARNING: This generates Floating Point code (`float*`).
        This is provided for baseline comparison/debugging. For the actual 
        "Deep Microcompression" bare-metal target, this layer is fused, and 
        this function is never called.
        """
        folded_weight, folded_bias = self.get_compression_parameters()

        input_row_size, input_col_size = input_shape[1:]

        input_channel_size = folded_weight.size(0)

        param_header, param_def = convert_tensor_to_bytes_var(folded_weight, f"{var_name}_folded_weight", for_arduino=for_arduino)
        layer_header = param_header
        layer_param_def = param_def

        param_header, param_def = convert_tensor_to_bytes_var(folded_bias, f"{var_name}_folded_bias", for_arduino=for_arduino)
        layer_header += param_header
        layer_param_def += param_def

        layer_def = (
            f"{self.__class__.__name__} {var_name}({input_channel_size}, {input_row_size}, {input_col_size}, "
            f"(float*){var_name}_folded_weight, (float*){var_name}_folded_bias);\n"
        )

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def