"""
@file flatten.py
@brief Flatten Layer with Structured Pruning Coordination.

    In the DMC pipeline, the Flatten layer is functionally simple (reshaping tensors)
    but architecturally critical for Structured Pruning (Section III-A).

    Role in Pruning:
    When a Convolutional channel is pruned, an entire 2D feature map ($H \times W$) 
    disappears. The Flatten layer must translate the "Surviving Channel Indices" 
    into a list of "Surviving Flat Element Indices" so the subsequent Linear layer 
    knows which input connections to remove.
"""

from typing import Union
import torch
from torch import nn

from .layer import Layer
from ..compressors import QuantizationScheme

from ..utils import (
    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2
)

class Flatten(Layer, nn.Flatten):
    """
    DMC-aware Flatten layer.
    
    Responsibilities:
    1.  Shape Transformation: Standard NCHW -> N(C*H*W) flattening.
    2.  Pruning Translation: Converts `keep_channel_mask` (spatial) to 
        `keep_neuron_mask` (flat) for dense matrix connectivity.
    3.  Quantization Pass-through: Preserves scale/zero-point metadata across
        the shape change without modifying values.
    """

    def __init__(self, *args, **kwargs):
        """Initialize Flatten layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """Forward pass preserving quantization context."""      
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
        """Coordinate channel pruning between layers by adjusting channel indices
        
        Args:
            sparsity: Target sparsity ratio (unused, maintained for interface consistency)
            keep_prev_channel_index: Channels to keep from previous layer
            is_output_layer: Flag if this is an output layer
            metric: Pruning metric (unused, maintained for interface consistency)
            
        Returns:
            Adjusted channel indices accounting for flatten operation
        """
        # Calculate number of elements per channel in original input
        channel_numel = input_shape[1:].numel()

        if is_output_layer:
            pass
            # Output layer doesn't prune, just pass through

        # Calculate start positions for each kept channel
        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(input_shape[0])
        start_positions = keep_prev_channel_index * channel_numel
        channel_elements_index = torch.arange(channel_numel).to(start_positions.device)

        # Generate indices for all elements in kept channels
        keep_current_channel_index = start_positions.view(-1, 1) + channel_elements_index

        return keep_current_channel_index.flatten()
    

    def get_prune_channel_possible_hyperparameters(self):
        return None
    
    def init_quantize(self, parameter_bitwidth, granularity, scheme, activation_bitwidth=None, previous_output_quantize = None):
        """
        Pass-through for quantization observers.
        
        Flattening does not change values, so the input scale/zero-point 
        is identical to the output scale/zero-point.
        """
        super().init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)

        return previous_output_quantize


    def get_quantize_possible_hyperparameters(self):
        return None
    

    def get_size_in_bits(self):
        return 0

    def get_compression_parameters(self):
        # Nothing to do
        pass


    def get_output_tensor_shape(self, input_shape: torch.Size):
        """Calculates flattened output shape."""
        return torch.Size((input_shape.numel(),)), torch.Size((input_shape.numel(),))
    

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
        input_size = torch.Size(input_shape).numel()
        scheme = None
        if self.is_quantized and "quantize" in self.__dict__["_dmc"]:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]

        if scheme != QuantizationScheme.STATIC:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
            layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
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
            
            layer_def = f"{self.__class__.__name__}_SQ {var_name}({input_size}, {quantize_property});\n"
            layer_header = f"extern {self.__class__.__name__}_SQ {var_name};\n\n"
        layer_param_def = ""

        return layer_header, layer_def, layer_param_def