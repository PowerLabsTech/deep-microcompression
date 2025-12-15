"""
@file linear.py
@brief Linear (Fully Connected) Layer for DMC Pipeline.

This class implements the "Dense" layers of the network (e.g., `linear_0`, `linear_1` in LeNet-5).
In the DMC experiments (Section IV-A), these layers often account for the majority of the 
model's storage footprint, making them prime targets for:
1.  Structured Pruning: Removing neurons reduces the matrix dimensions physically.
2.  Bit-Packing: Compressing the large weight matrices into 4-bit/2-bit streams.
"""
__all__ = [
    "Linear"
]

from typing import Optional, Tuple, Union
from functools import partial

import torch
from torch import nn

from .layer import Layer
from ..compressors import (
    Prune_Channel, 
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity
)

from ..utils import (
    convert_tensor_to_bytes_var,
    get_size_in_bits,

    STATIC_BIAS_BITWDHT,
)

class Linear(Layer, nn.Linear):
    """
    DMC-Optimized Linear Layer.
    
    Supports:
    - Sensitivity Analysis: exposing hyperparameter ranges for pruning search.
    - Dependency Propagation: pruning input weights based on previous layer's mask.
    - Hardware-Aware Packing: exporting weights in packed `int8` format.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Linear layer with standard PyTorch parameters"""
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or fake quantized)
            
        Returns:
            Output tensor according to current quantization mode
        """
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                # Structured Pruning: Mask out removed neurons/connections
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)

            # Quantization: Simulate integer precision loss
            if self.is_quantized:
                # Note: Input is already quantized by previous layer's output_quantize
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        output = nn.functional.linear(input, weight, bias)
        
        if self.is_compressed:
            # Rescalling
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)

                    # print(self.input_quantize.zero_point, self)

        return output
    

    @torch.no_grad()
    def init_prune_channel(
            self, 
            sparsity: Union[float, int], 
            keep_prev_channel_index: Optional[torch.Tensor], 
            input_shape: torch.Size,
            is_output_layer: bool = False, 
            metric: str = "l2"
        ) -> Optional[torch.Tensor]:
        """
        Executes Structured Pruning logic (Section III-A).
        
        For a Linear layer, "Channel Pruning" equates to "Neuron Pruning".
        
        Operations:
        1.  Input Dimension Reduction: Uses `keep_prev_channel_index` (from the 
            previous Flatten/Linear/Conv layer) to physically remove input columns.
        2.  Output Dimension Reduction: Calculates neuron importance (L2 norm) 
            and removes the least important rows (neurons).
        
        Args:
            sparsity: Amount to prune (float 0.0-1.0 or integer count).
            keep_prev_channel_index: Indices of valid inputs.
            is_output_layer: If True, output neurons (classes) are NEVER pruned.
            metric: Importance criteria ('l2' or 'l1').
            
        Returns:
            Indices of kept output neurons (to be passed to next layer).
        """
        # Convert sparsity ratio to integer count of Pruned Channels
        if isinstance(sparsity, float):
            sparsity = min(max(0., sparsity), 1.)
            sparsity = int(sparsity * self.out_features)
        elif isinstance(sparsity, int): pass
        else:
            raise ValueError(f"sparsity must be of type int or float, got type {type(sparsity)}")

        sparsity = min(max(0, sparsity), self.out_features-1)
        density = self.out_features - sparsity

        # Handle Output Neurons (Rows)
        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_features)

        if is_output_layer:
            # Skip pruning for output layer
            keep_current_channel_index = torch.arange(self.out_features)
        else:

            # Calculate channel importance
            importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
            channel_importance = importance.sum(dim=[1])
            keep_current_channel_index = torch.sort(torch.topk(channel_importance, density, dim=0).indices).values
               
        # Store Indices
        keep_prev_channel_index = keep_prev_channel_index.to(self.weight.device)
        keep_current_channel_index = keep_current_channel_index.to(self.weight.device)

        setattr(self, "weight_prune_channel", Prune_Channel(
            layer=self, keep_current_channel_index=keep_current_channel_index, keep_prev_channel_index=keep_prev_channel_index
        ))

        if self.bias is not None:
            setattr(self, "bias_prune_channel", Prune_Channel(
                layer=self, keep_current_channel_index=keep_current_channel_index
            ))
        return keep_current_channel_index


    def get_prune_channel_possible_hyperparameters(self):
        """
        Returns search space for Sensitivity Analysis.
        Allows testing every possible neuron count from 1 to N.
        """
        return range(self.out_features)

    @torch.no_grad()
    def init_quantize(
        self, 
        bitwidth: int, 
        scheme: QuantizationScheme, 
        granularity: QuantizationGranularity, 
        previous_output_quantize: Optional[Quantize] = None
    ):
        """
        Sets up Quantization Observers.
        
        Key Logic:
        - Weights: Symmetric Quantization (Int8/4/2).
        - Inputs/Outputs: Asymmetric (UInt8/Int8).
        - Bias: 32-bit Symmetric. Scale is constrained to `input_scale * weight_scale`
          to allow efficient integer MAC operations.
        """
        # Weight Quantizer
        if not self.is_pruned_channel:
            setattr(self, "weight_quantize", Quantize(
                self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC
            ))
        else:
            setattr(self, "weight_quantize", Quantize(
                self, bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, prune_channel=self.weight_prune_channel
            ))

        # Input/Output Quantizers (Required for Static Scheme)
        if scheme == QuantizationScheme.STATIC:
            assert previous_output_quantize is not None, "Pass a quantizer for the input, it is usually from the preceeding layer."
            setattr(self, "input_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC, base=[previous_output_quantize]
            ))
            setattr(self, "output_quantize", Quantize(
                self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
            ))

        # Bias Quantizer
        if self.bias is not None:
            if not self.is_pruned_channel:
                if scheme == QuantizationScheme.STATIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, STATIC_BIAS_BITWDHT, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize, self.input_quantize]
                    ))
            else:
                if scheme == QuantizationScheme.STATIC:
                    setattr(self, "bias_quantize", Quantize(
                        self, STATIC_BIAS_BITWDHT, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, base=[self.weight_quantize, self.input_quantize], prune_channel=self.bias_prune_channel
                    ))

        # calibration
        if scheme == QuantizationScheme.DYNAMIC:
            self.weight_quantize.update_parameters(self.weight) 

        if hasattr(self, "output_quantize"):
            return self.output_quantize 
        return None

    @torch.no_grad()
    def get_size_in_bits(self):  
        """
        Calculates compressed model size.
        Includes overhead for quantization parameters (Scales/Zero-points).
        """
        weight, bias = self.get_compression_parameters()

        is_packed = False
        weight_bitwidth = None
        
        size = 0

        bias_bitwidth = None
        if self.is_quantized:
            is_packed = True
            weight_bitwidth = self.weight_quantize.bitwidth
            if self.bias is not None and hasattr(self, "bias_quantize"):
                bias_bitwidth = self.bias_quantize.bitwidth

            # Add storage cost for Quantization Metadata (Scales/ZP)
            if self.weight_quantize.scheme == QuantizationScheme.DYNAMIC:
                size += get_size_in_bits(self.weight_quantize.scale)
            elif self.weight_quantize.scheme == QuantizationScheme.STATIC:
                size += get_size_in_bits(self.output_quantize.scale)
                size += get_size_in_bits(self.output_quantize.zero_point)
                size += get_size_in_bits(self.input_quantize.zero_point)

                if self.bias is not None:
                    bias_scale = self.bias_quantize.scale
                else:
                    bias_scale = self.input_quantize.scale * self.weight_quantize.scale
                size += get_size_in_bits(bias_scale)
        
        # Add storage cost for Weights and Biases (Potentially Bit-Packed)
        size += get_size_in_bits(weight, is_packed=is_packed, bitwidth=weight_bitwidth)
        if self.bias is not None:
            size += get_size_in_bits(bias, is_packed=is_packed, bitwidth=bias_bitwidth)  

        return size



    @torch.no_grad()
    def get_compression_parameters(self) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns the final hard-pruned and hard-quantized tensors."""
        weight = self.weight
        bias = self.bias

        if self.is_compressed:

            # Hard Pruning (Slicing)
            if self.is_pruned_channel:
                weight = self.weight_prune_channel.apply(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel.apply(bias)

            # Hard Quantization (Float -> Int)
            if self.is_quantized:
                    weight = self.weight_quantize.apply(weight)
                    if self.bias is not None and hasattr(self, "bias_quantize"):
                        bias = self.bias_quantize.apply(bias)
        
        return weight, bias


    def get_output_tensor_shape(self, input_shape) -> tuple[torch.Size, torch.Size]:
        """Calculates output shape for memory planning."""
        out_features, _ = self.get_compression_parameters()[0].size()
        return torch.Size((out_features,)), torch.Size((out_features,))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """
        Generates C code for deployment (Stage 3).

        Args:
            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if needed
        
        Produces one of three C constructor variants:
        1.  Float: Standard `float*` pointers (Baseline).
        2.  Dynamic: `int8_t` weights, `float` bias/scales (Reference).
        3.  Static (DMC): Fully integer. `int8_t` weights, `int32_t` bias.
            Includes `zero_point` parameters for hardware-aware unpacking.
            
        Returns:
            (Header String, Definition String, Parameter Blob String)
        """

        weight, bias = self.get_compression_parameters()
        
        output_feature_size, input_feature_size = weight.size()

        weight_bitwidth = None
        if self.is_quantized:
            weight_bitwidth = self.weight_quantize.bitwidth

        # Convert weights to C representation
        param_header, param_def = convert_tensor_to_bytes_var(
            weight, 
            f"{var_name}_weight", 
            weight_bitwidth,
            for_arduino=for_arduino
        )   
        layer_header = param_header
        layer_param_def = param_def

        if self.bias is not None:
            bias_bitwidth = None
            if self.is_quantized and hasattr(self, "bias_quantize"):
                bias_bitwidth = self.bias_quantize.bitwidth
                # print(bias.dtype, "in linear bias dtype")
            param_header, param_def = convert_tensor_to_bytes_var(
                bias, 
                f"{var_name}_bias",
                bias_bitwidth,
                for_arduino=for_arduino
            )
            layer_header += param_header
            layer_param_def += param_def
            # print("----------->utilis after", param_def)

        scheme = None
        if self.is_quantized:
            scheme = self.weight_quantize.scheme

        if scheme is None or scheme == QuantizationScheme.NONE:
            if self.bias is not None:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, (float*){var_name}_bias);\n"
            else:
                layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (float*){var_name}_weight, nullptr);\n"
                
        elif scheme == QuantizationScheme.DYNAMIC:
            granularity = self.weight_quantize.granularity

            if self.bias is not None:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, (float*){var_name}_bias, *(float*){var_name}_weight_scale);\n"   
                else:
                    layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, (float*){var_name}_bias, (float*){var_name}_weight_scale);\n"
            else:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, nullptr, *(float*){var_name}_weight_scale);\n"   
                else:
                    layer_def = f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, nullptr, (float*){var_name}_weight_scale);\n"


            param_header, param_def = convert_tensor_to_bytes_var(
                                        self.weight_quantize.scale,
                                        f"{var_name}_weight_scale"
                                    )
            layer_header += param_header
            layer_param_def += param_def
            # print("----------->utilis after", layer_param_def)
            
        elif scheme == QuantizationScheme.STATIC:
            granularity = self.weight_quantize.granularity

            if self.bias is not None:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, (int32_t*){var_name}_bias, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, *(int8_t*){var_name}_input_zero_point, "
                        f"*(float*){var_name}_bias_scale);\n"
                    ) 
                else:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, (int32_t*){var_name}_bias, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, *(int8_t*){var_name}_input_zero_point, "
                        f"(float*){var_name}_bias_scale);\n"
                    )
            else:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, nullptr, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, *(int8_t*){var_name}_input_zero_point, "
                        f"*(float*){var_name}_bias_scale);\n"
                    ) 
                else:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({output_feature_size}, {input_feature_size}, (int8_t*){var_name}_weight, nullptr, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, *(int8_t*){var_name}_input_zero_point, "
                        f"(float*){var_name}_bias_scale);\n"
                    )

            param_header, param_def = convert_tensor_to_bytes_var(
                self.output_quantize.scale, 
                f"{var_name}_output_scale"
            )
            layer_header += param_header
            layer_param_def += param_def

            param_header, param_def = convert_tensor_to_bytes_var(
                self.output_quantize.zero_point, 
                f"{var_name}_output_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def

            param_header, param_def = convert_tensor_to_bytes_var(
                self.input_quantize.zero_point, 
                f"{var_name}_input_zero_point"
            )
            layer_header += param_header
            layer_param_def += param_def

            if self.bias is not None:
                bias_scale = self.bias_quantize.scale
            else:
                bias_scale = self.input_quantize.scale * self.weight_quantize.scale
            param_header, param_def = convert_tensor_to_bytes_var(
                bias_scale,
                f"{var_name}_bias_scale"
            )
            layer_header += param_header
            layer_param_def += param_def


        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        return layer_header, layer_def, layer_param_def