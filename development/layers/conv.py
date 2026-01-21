"""
@file conv.py
@brief Convolutional Layer for DMC Pipeline.

This class implements the core convolutional logic for the compression pipeline.
It handles:
1.  Structured Pruning: Calculating filter importance and generating masks.
2.  Quantization: Managing observers for Weights, Inputs, and Biases.
3.  Code Generation: Exporting parameters to the hardware-aware C library.
"""

__all__ = [
    "Conv2d"
]

from typing import Optional, Tuple, Union

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


class Conv2d(Layer, nn.Conv2d):
    """
    DMC-Optimized Conv2d Layer.
    
    Supports:
    - Sensitivity Analysis: exposing hyperparameter ranges for pruning search.
    - Dependency Propagation: pruning input weights based on previous layer's mask.
    - Hardware-Aware Packing: exporting weights in packed `int8` format.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Conv2d layer with standard PyTorch parameters"""
        # Extract custom padding logic (Left, Right, Top, Bottom)
        self.pad = kwargs.pop("pad", (0, 0, 0, 0))
        assert len(self.pad) == 4, f"pad {self.pad} is invalid, pad should be of (pad_left, pad_right, pad_top, pad_bottom)"
        
        groups = kwargs.get("groups", 1)

        # Enforce usage of explicit pad instead of built-in padding arg
        if "padding" in kwargs:
            assert kwargs["padding"] == 0, "Use pad instead of padding to pad input"
        else:
            kwargs["padding"] = 0

        super().__init__(*args, **kwargs)

        # Constraint: DMC pruning currently supports standard (groups=1) or Depthwise (groups=C)
        assert groups == 1 or groups == self.out_channels, \
            "DMC currently supports only Standard (groups=1) or Depthwise (groups=out_channels) convolution."
        
        
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization awareness
        
        Args:
            input: Input tensor (float or quantized)
            
        Returns:
            Output tensor after convolution with quantization if enabled
        """
        # Perform convolution with appropriate padding
        
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            # Structured Pruning
            # Apply binary masks to filters
            if self.is_pruned_channel:
                
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)
                    
            # Quantization
            # Simulate low-bitwidth integer arithmetic
            if self.is_quantized:
                # Note: Input is already quantized by previous layer's output_quantize
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        # Apply explicit padding
        input =  nn.functional.pad(input, self.pad, "constant", 0) 
        output = nn.functional.conv2d(
            input, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    # Re-scale 32-bit accumulator to target bitwidth (e.g., 8-bit)
                    output = self.output_quantize(output)

        return output


    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        """
        Executes Structured Pruning logic.
        
        This method calculates the importance of each convolutional filter and
        determines which ones to keep.
        
        Handling Dependencies:
        1. Standard Conv: We can prune filters (Output Channels) freely based on 
           metric (L2). Input channels are pruned based on `keep_prev_channel_index`.
        2. Depthwise Conv: Input and Output channels are 1:1 coupled. We cannot 
           prune them independently. We must respect the previous layer's decision.
        
        Args:
            sparsity: Target removal ratio (0.0 - 1.0).
            keep_prev_channel_index: Indices of valid input channels from previous layer.
            metric: 'l2' (Magnitude) or 'l1'.
            
        Returns:
            Indices of kept output channels.
        """
        # Validate Grouped Convolution constraints
        assert(self.groups == 1 or self.groups == self.out_channels), \
            "Grouped convolution pruning not fully supported."
        
        # Convert sparsity float to integer count
        if isinstance(sparsity, float):
            sparsity = min(max(0., sparsity), 1.)
            sparsity = int(sparsity * self.out_channels)
        elif isinstance(sparsity, int): 
            pass
        else:
            raise ValueError(f"Sparsity type error: {type(sparsity)}")
        
        sparsity = min(max(0, sparsity), self.out_channels-1)
        density = self.out_channels - sparsity

        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(self.in_channels)
        if self.groups == self.out_channels:

            keep_prev_channel_index_temp = keep_prev_channel_index
            keep_prev_channel_index = torch.arange(1)

            if is_output_layer:
                keep_current_channel_index = torch.arange(self.out_channels)
            else:
                keep_current_channel_index = keep_prev_channel_index_temp
        else:

            if is_output_layer:
                keep_current_channel_index = torch.arange(self.out_channels)

            else:
                # Select top-k neurons to keep
                importance = self.weight.pow(2) if metric == "l2" else self.weight.abs()
                channel_importance = importance.sum(dim=[1, 2, 3])
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
        """Returns valid channel counts for Search Phase."""
        if self.groups == self.out_channels:
            return None
        return range(self.out_channels)


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
        
        Logic:
        1. Weights: Symmetric (Int8/4/2).
        2. Inputs/Outputs: Asymmetric (UInt8/Int8) - Required for Static.
        3. Bias: 32-bit Symmetric, scaled by (Input_Scale * Weight_Scale).
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

        # Activation Quantizers (Static Mode)
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
    def get_size_in_bits(self) -> int:
        """Calculates total storage footprint in bits."""
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

            # Add metadata overhead
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
        C_in, H_in, W_in = input_shape
        
        # Unpack parameters (handle both int and tuple)
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        # kH, kW = _pair(self.kernel_size)
        C_out, _, kH, kW = self.get_compression_parameters()[0].size()
            
        sH, sW = _pair(self.stride)
        dH, dW = _pair(self.dilation)

        pW = self.pad[0] + self.pad[1]
        pH = self.pad[2] + self.pad[3]
        
        H_out = ((H_in +  pH - dH * (kH - 1) - 1) // sH) + 1
        W_out = ((W_in +  pW - dW * (kW - 1) - 1) // sW) + 1
        
        return torch.Size((C_in, H_in +  pH, W_in +  pW)), torch.Size((C_out, H_out, W_out))
    

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """
        Generates C code for deployment.

        Args:
            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if needed
        
        Key DMC Features:
        - Bit-Packing: Calls `convert_tensor_to_bytes_var` which packs 
          low-bitwidth weights into `int8` arrays.
        - Hardware Types: Generates `int8_t` weight buffers and `int32_t` 
          bias buffers for Static Quantization.
        """

        weight, bias = self.get_compression_parameters()

        input_channel_size, input_row_size, input_col_size = input_shape

        output_channel_size, _,\
        kernel_row_size, kernel_col_size = weight.size()
        stride_row, stride_col = self.stride
        pad = self.pad

        if self.groups == self.out_channels:
            groups = input_channel_size
        else:
            groups = self.groups

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
            param_header, param_def = convert_tensor_to_bytes_var(
                bias, 
                f"{var_name}_bias",
                bias_bitwidth,
                for_arduino=for_arduino
            )
            layer_header += param_header
            layer_param_def += param_def

        scheme = None
        if self.is_quantized:
            scheme = self.weight_quantize.scheme

        if scheme is None or scheme == QuantizationScheme.NONE:
            if self.bias is not None:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(float*){var_name}_weight, (float*){var_name}_bias);\n"
                )
            else:
                layer_def = (
                    f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                    f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                    f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                    "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                    f"(float*){var_name}_weight, nullptr);\n"
                )
        elif scheme == QuantizationScheme.DYNAMIC:
            granularity = self.weight_quantize.granularity
            
            if self.bias is not None:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, (float*){var_name}_bias,  *(float*){var_name}_weight_scale);\n"
                    )  
                else:        
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, (float*){var_name}_bias,  (float*){var_name}_weight_scale);\n"
                    )
            else:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, nullptr,  *(float*){var_name}_weight_scale);\n"
                    )  
                else:        
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, nullptr,  (float*){var_name}_weight_scale);\n"
                    )
            param_header, param_def = convert_tensor_to_bytes_var(
                                        self.weight_quantize.scale,
                                        f"{var_name}_weight_scale"
                                    )
            layer_header += param_header
            layer_param_def += param_def

        elif scheme == QuantizationScheme.STATIC:
            granularity = self.weight_quantize.granularity

            if self.bias is not None:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, (int32_t*){var_name}_bias, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, "
                        f"*(int8_t*){var_name}_input_zero_point, *(float*){var_name}_bias_scale);\n"
                    )
                else:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, (int32_t*){var_name}_bias, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, "
                        f"*(int8_t*){var_name}_input_zero_point, (float*){var_name}_bias_scale);\n"
                    )
            else:
                if granularity == QuantizationGranularity.PER_TENSOR:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, nullptr, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, "
                        f"*(int8_t*){var_name}_input_zero_point, *(float*){var_name}_bias_scale);\n"
                    )
                else:
                    layer_def = (
                        f"{self.__class__.__name__} {var_name}({input_channel_size}, "
                        f"{input_row_size}, {input_col_size}, {output_channel_size}, "
                        f"{kernel_row_size}, {kernel_col_size}, {stride_row}, {stride_col}, "
                        "{" f"{pad[0]}, {pad[1]}, {pad[2]}, {pad[3]}" "}, " f"{groups}, "
                        f"(int8_t*){var_name}_weight, nullptr, "
                        f"*(float*){var_name}_output_scale, *(int8_t*){var_name}_output_zero_point, "
                        f"*(int8_t*){var_name}_input_zero_point, (float*){var_name}_bias_scale);\n"
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

