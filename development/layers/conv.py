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

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .layer import Layer
from ..compressors import (
    Prune_Channel,
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
    QuantizationBitWidthError,
    QuantizationGranularityError,
)

from ..utils import (
    convert_tensor_to_bytes_var,
    get_size_in_bits,

    STATIC_BIAS_BITWDHT,

    PER_TENSOR,
    PER_CHANNEL,
    PARAMETER_BITWIDTH_2,
    PARAMETER_BITWIDTH_4,
    PARAMETER_BITWIDTH_8,

    ACTIVATION_BITWIDTH_2,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_8,

    INPUT_ACTIVATION_BITWIDTH_2,
    INPUT_ACTIVATION_BITWIDTH_4,
    INPUT_ACTIVATION_BITWIDTH_8,

    UINT8_T,
    UINT16_T,
    INT8_T,
    FLOAT_T,
    VOID_PTR,
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
        # Enforce usage of explicit pad instead of built-in padding arg

        super().__init__(*args, **kwargs)

        # Constraint: DMC pruning currently supports standard (groups=1) or Depthwise (groups=C)
        assert not self.is_grouped(), \
            "DMC currently supports only Standard (groups=1) or Depthwise (in_channels=groups=out_channels) convolution."
        
        
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


    def is_depthwise(self):
        return self.in_channels == self.groups and self.in_channels == self.out_channels


    def is_grouped(self):
        if self.groups == 1:
            return False
        return not self.is_depthwise()


    @torch.no_grad()
    def init_prune_channel(
        self, 
        sparsity: float, 
        input_shape: torch.Size,
        keep_prev_channel_index:Optional[torch.Tensor], 
        keep_current_channel_index:Optional[torch.Tensor],
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
        assert not self.is_grouped(), \
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

        # For depthwise convolution, the second dimension is 1 as all
        # filters match with the input channel
        if self.is_depthwise():

            keep_prev_channel_index_temp = keep_prev_channel_index
            keep_prev_channel_index = torch.arange(1)

            if keep_current_channel_index is None:
                if is_output_layer:
                    keep_current_channel_index = torch.arange(self.out_channels)
                else:
                    keep_current_channel_index = keep_prev_channel_index_temp
        else:
            if keep_current_channel_index is None:

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
        if self.is_depthwise():
            return None
        return range(self.out_channels)


    @torch.no_grad()
    def init_quantize(
        self, 
        parameter_bitwidth: int, 
        granularity: QuantizationGranularity, 
        scheme: QuantizationScheme,
        activation_bitwidth:Optional[int]=None,
        previous_output_quantize: Optional[Quantize] = None,
        current_output_quantize: Optional[Quantize] = None,
    ):
        """
        Sets up Quantization Observers.
        
        Logic:
        1. Weights: Symmetric (Int8/4/2).
        2. Inputs/Outputs: Asymmetric (UInt8/Int8) - Required for Static.
        3. Bias: 32-bit Symmetric, scaled by (Input_Scale * Weight_Scale).
        """
        super().init_quantize(
            parameter_bitwidth, granularity, scheme, activation_bitwidth,
            previous_output_quantize, change_quantization_scale=True
        )
        activation_bitwidth = self.__dict__["_dmc"]["quantize"]["activation_bitwidth"]

        # Weight Quantizer
        if not self.is_pruned_channel:
            setattr(self, "weight_quantize", Quantize(
                self, parameter_bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC
            ))
        else:
            setattr(self, "weight_quantize", Quantize(
                self, parameter_bitwidth, scheme, granularity, scale_type=QuantizationScaleType.SYMMETRIC, prune_channel=self.weight_prune_channel
            ))

        # Activation Quantizers (Static Mode)
        if scheme == QuantizationScheme.STATIC:
            assert activation_bitwidth is not None, "Pass an activation bitwidth when doing static quantization"
            assert previous_output_quantize is not None, "Pass a quantizer for the input, it is usually from the preceeding layer."
            setattr(self, "input_quantize", Quantize(
                self, activation_bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC, base=[previous_output_quantize]
            ))
            # Fixing the output quantizer to be that passed,likely due to it being in a branch layer
            if current_output_quantize is None:
                setattr(self, "output_quantize", Quantize(
                    self, activation_bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
                ))
            else:
                setattr(self, "output_quantize", Quantize(
                    self, activation_bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC, base=[current_output_quantize]
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

    def get_quantize_possible_hyperparameters(self, scheme:QuantizationScheme=QuantizationScheme.STATIC):
        if scheme == QuantizationScheme.STATIC:
            return {
                "parameter_bitwidth": [8, 4, 2],
                "activation_bitwidth": [8, 4, 2],
                "granularity": [QuantizationGranularity.PER_TENSOR, QuantizationGranularity.PER_CHANNEL],
            }
        return {
            "parameter_bitwidth": [8, 4, 2],
            "activation_bitwidth": [8, 4, 2],
            "granularity": [QuantizationGranularity.PER_TENSOR, QuantizationGranularity.PER_CHANNEL],
        }

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
            if self.bias is not None:
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
    

    def get_workspace_debt(self, input_shape):
        if self.in_channels >= self.out_channels:
            return 0

        C_in, H_p, W_p = self.get_padded_input_tensor_shape(input_shape)
        C_out, H_out, W_out = self.get_output_tensor_shape(input_shape)

        assert C_in == self.in_channels, (
            "The channel of the input shape does not match in_channels, "
            f"expected {self.in_channels} but got {C_in}"
        )
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        sH, sW = _pair(self.stride)

        # Paper Theorem 3: D(y,x) = - x*α - y*β
        alpha = C_in * sW - C_out
        beta  = W_p * sH * C_in - C_out * W_out
        D = lambda y, x: - x * alpha - y * beta

        max_D = 0
        for (y, x) in [(0, 0), (0, W_out-1), (H_out-1, 0), (H_out-1, W_out-1)]:
            max_D = max(max_D, D(y, x))
        return max_D


    def get_workspace_size(
        self, input_shape, data_per_byte,
        include_locals=False, include_runtime=False, ptr_size=2
    ) -> int:
        output_channel_per_group = self.out_channels // self.groups
        base = math.ceil(self.get_padded_input_tensor_shape(input_shape).numel() / data_per_byte)
        base += math.ceil(self.get_workspace_debt(input_shape) / data_per_byte)
        base += math.ceil(output_channel_per_group / data_per_byte)
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized:
            scheme = self.weight_quantize.scheme
        if scheme == QuantizationScheme.STATIC:
            # 4×u16 + 7×u8 from conv header + f32 output_scale + i8 output_zp + i8 input_zp + u8 property
            locals_size  = 22
            # 3 Flash ptrs + 2 workspace alias ptrs + computed(20) + 4 fn ptrs + loops(15) + extras(scale_index u8+val i32=5)
            runtime_size = 40 + 9 * ptr_size
        elif scheme == QuantizationScheme.DYNAMIC:
            # 4×u16 + 7×u8 from conv header + u8 quantize_property
            locals_size  = 16
            # 3 Flash ptrs + 2 workspace alias ptrs + computed(12) + 1 fn ptr + loops(15) + scale_index u8
            runtime_size = 28 + 6 * ptr_size
        else:
            # 4×u16 + 7×u8 from conv header
            locals_size  = 15
            # 2 Flash ptrs + 2 workspace alias ptrs + computed(12) + loops(15)
            runtime_size = 27 + 4 * ptr_size
        return base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0)


    def get_padded_input_tensor_shape(self, input_shape) -> torch.Size:
        """Calculates input shape after padding"""
        C_in, H_in, W_in = input_shape
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        pH, pW = _pair(self.padding)
        H_out = H_in + 2 * pH
        W_out = W_in + 2 * pW
        
        return torch.Size((C_in, H_out, W_out))
    


    def get_output_tensor_shape(self, input_shape) -> torch.Size:
        """Calculates output shape for memory planning."""
        C_in, H_in, W_in = self.get_padded_input_tensor_shape(input_shape)
        
        # Unpack parameters (handle both int and tuple)
        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        
        # kH, kW = _pair(self.kernel_size)
        C_out, _, kH, kW = self.get_compression_parameters()[0].size()
            
        sH, sW = _pair(self.stride)
        dH, dW = _pair(self.dilation)
        
        H_out = ((H_in - dH * (kH - 1) - 1) // sH) + 1
        W_out = ((W_in - dW * (kW - 1) - 1) // sW) + 1
        
        return torch.Size((C_out, H_out, W_out))


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

 
        # Reorder from OIHW (PyTorch) to OHWI for HWC activation layout
        if weight.dim() == 4:
            weight = weight.permute(0, 2, 3, 1).contiguous()

        input_channel_size, input_row_size, input_col_size = input_shape

        output_channel_size, kernel_row_size, kernel_col_size, _ = weight.size()
        stride_row, stride_col = self.stride

        # For dephtwise convolution, so the groups matchs with the input 
        # channel in a case where some layers are pruned off.
        if self.is_depthwise():
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

        def _pair(x): return x if isinstance(x, tuple) else (x, x)
        padding_row, padding_col = _pair(self.padding)

        if scheme is None or scheme == QuantizationScheme.NONE:
            bias_ptr = f"(void*){var_name}_bias" if self.bias is not None else "nullptr"
            params_info = [
                (UINT16_T, "input_channel", str(input_channel_size)),
                (UINT16_T, "input_row",     str(input_row_size)),
                (UINT16_T, "input_col",     str(input_col_size)),
                (UINT16_T, "output_channel", str(output_channel_size)),
                (UINT8_T,  "kernel_row",    str(kernel_row_size)),
                (UINT8_T,  "kernel_col",    str(kernel_col_size)),
                (UINT8_T,  "padding_row",   str(padding_row)),
                (UINT8_T,  "padding_col",   str(padding_col)),
                (UINT8_T,  "stride_row",    str(stride_row)),
                (UINT8_T,  "stride_col",    str(stride_col)),
                (UINT8_T,  "groups",        str(groups)),
                (UINT8_T,  "_align",        "0"),
                (VOID_PTR, "weight",        f"(void*){var_name}_weight"),
                (VOID_PTR, "bias",          bias_ptr),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.NONE, for_arduino)
            layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
            
        elif scheme == QuantizationScheme.DYNAMIC:

            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
            granularity = self.__dict__["_dmc"]["quantize"]["granularity"]
            parameter_bitwidth = self.__dict__["_dmc"]["quantize"]["parameter_bitwidth"]

            quantize_property = ""

            if granularity == QuantizationGranularity.PER_TENSOR:
                quantize_property += PER_TENSOR
            elif granularity == QuantizationGranularity.PER_CHANNEL:
                quantize_property += PER_CHANNEL
            else:
                raise QuantizationGranularityError(granularity)

            quantize_property += "_"

            if parameter_bitwidth == 8:
                quantize_property += PARAMETER_BITWIDTH_8
            elif parameter_bitwidth == 4:
                quantize_property += PARAMETER_BITWIDTH_4
            elif parameter_bitwidth == 2:
                quantize_property += PARAMETER_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(parameter_bitwidth)

            bias_ptr = f"(void*){var_name}_bias" if self.bias is not None else "nullptr"
            params_info = [
                (UINT16_T, "input_channel",    str(input_channel_size)),
                (UINT16_T, "input_row",        str(input_row_size)),
                (UINT16_T, "input_col",        str(input_col_size)),
                (UINT16_T, "output_channel",   str(output_channel_size)),
                (UINT8_T,  "kernel_row",       str(kernel_row_size)),
                (UINT8_T,  "kernel_col",       str(kernel_col_size)),
                (UINT8_T,  "padding_row",      str(padding_row)),
                (UINT8_T,  "padding_col",      str(padding_col)),
                (UINT8_T,  "stride_row",       str(stride_row)),
                (UINT8_T,  "stride_col",       str(stride_col)),
                (UINT8_T,  "groups",           str(groups)),
                (UINT8_T,  "_align",           "0"),
                (VOID_PTR, "weight",           f"(void*){var_name}_weight"),
                (VOID_PTR, "bias",             bias_ptr),
                (VOID_PTR, "weight_scale",     f"(void*){var_name}_weight_scale"),
                (UINT8_T,  "quantize_property", quantize_property),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.DYNAMIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_DQ {var_name};\n\n"

            param_header, param_def = convert_tensor_to_bytes_var(
                self.weight_quantize.scale,
                f"{var_name}_weight_scale",
                for_arduino=for_arduino
            )
            layer_header += param_header
            layer_param_def += param_def

        elif scheme == QuantizationScheme.STATIC:

            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
            granularity = self.__dict__["_dmc"]["quantize"]["granularity"]
            parameter_bitwidth = self.__dict__["_dmc"]["quantize"]["parameter_bitwidth"]
            activation_bitwidth = self.__dict__["_dmc"]["quantize"]["activation_bitwidth"]
            input_activation_bitwidth = self.__dict__["_dmc"]["quantize"]["input_activation_bitwidth"]

            quantize_property = ""

            if input_activation_bitwidth == 8:
                quantize_property += INPUT_ACTIVATION_BITWIDTH_8
            elif input_activation_bitwidth == 4:
                quantize_property += INPUT_ACTIVATION_BITWIDTH_4
            elif input_activation_bitwidth == 2:
                quantize_property += INPUT_ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(input_activation_bitwidth)

            quantize_property += "_"

            if granularity == QuantizationGranularity.PER_TENSOR:
                quantize_property += PER_TENSOR
            elif granularity == QuantizationGranularity.PER_CHANNEL:
                quantize_property += PER_CHANNEL
            else:
                raise QuantizationGranularityError(granularity)

            quantize_property += "_"

            if activation_bitwidth == 8:
                quantize_property += ACTIVATION_BITWIDTH_8
            elif activation_bitwidth == 4:
                quantize_property += ACTIVATION_BITWIDTH_4
            elif activation_bitwidth == 2:
                quantize_property += ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(activation_bitwidth)

            quantize_property += "_"

            if parameter_bitwidth == 8:
                quantize_property += PARAMETER_BITWIDTH_8
            elif parameter_bitwidth == 4:
                quantize_property += PARAMETER_BITWIDTH_4
            elif parameter_bitwidth == 2:
                quantize_property += PARAMETER_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(parameter_bitwidth)

            output_scale_val      = float(self.output_quantize.scale.item())
            output_zero_point_val = int(self.output_quantize.zero_point.item())
            input_zero_point_val  = int(self.input_quantize.zero_point.item())
            bias_ptr = f"(void*){var_name}_bias" if self.bias is not None else "nullptr"
            params_info = [
                (UINT16_T, "input_channel",      str(input_channel_size)),
                (UINT16_T, "input_row",          str(input_row_size)),
                (UINT16_T, "input_col",          str(input_col_size)),
                (UINT16_T, "output_channel",     str(output_channel_size)),
                (UINT8_T,  "kernel_row",         str(kernel_row_size)),
                (UINT8_T,  "kernel_col",         str(kernel_col_size)),
                (UINT8_T,  "padding_row",        str(padding_row)),
                (UINT8_T,  "padding_col",        str(padding_col)),
                (UINT8_T,  "stride_row",         str(stride_row)),
                (UINT8_T,  "stride_col",         str(stride_col)),
                (UINT8_T,  "groups",             str(groups)),
                (UINT8_T,  "_align",             "0"),
                (VOID_PTR, "weight",             f"(void*){var_name}_weight"),
                (VOID_PTR, "bias",               bias_ptr),
                (VOID_PTR, "bias_scale",         f"(void*){var_name}_bias_scale"),
                (FLOAT_T,  "output_scale",       f"{output_scale_val:.9g}f"),
                (INT8_T,   "output_zero_point", str(output_zero_point_val)),
                (INT8_T,   "input_zero_point",  str(input_zero_point_val)),
                (UINT8_T,  "quantize_property",  quantize_property),
            ]
            layer_def = self.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

            if self.bias is not None:
                bias_scale = self.bias_quantize.scale
            else:
                bias_scale = self.input_quantize.scale * self.weight_quantize.scale

            # removing scales for channels that have been pruned away
            if self.is_pruned_channel and granularity == QuantizationGranularity.PER_CHANNEL:
                bias_scale = self.bias_prune_channel.apply(bias_scale)
        
            param_header, param_def = convert_tensor_to_bytes_var(
                bias_scale,
                f"{var_name}_bias_scale",
                for_arduino=for_arduino
            )
            layer_header += param_header
            layer_param_def += param_def
   

        return layer_header, layer_def, layer_param_def

