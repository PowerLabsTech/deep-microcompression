"""
@file utils.py
@brief Utility functions for quantization, dequantization, and byte conversion
"""

import struct
import math
from enum import Enum, auto

from typing import Any, Tuple, Optional

import torch
from torch import nn


# Quantization type constants

STATIC_BIAS_BITWDHT = 32


INT8_BYTE_PER_LINE = 16
FLOAT32_BYTE_PER_LINE = 4
INT32_BYTE_PER_LINE = 4

PER_TENSOR = "PER_TENSOR"
PER_CHANNEL = "PER_CHANNEL"

ACTIVATION_BITWIDTH_8 = "A8"
ACTIVATION_BITWIDTH_4 = "A4"
ACTIVATION_BITWIDTH_2 = "A2"

PARAMETER_BITWIDTH_8 = "P8"
PARAMETER_BITWIDTH_4 = "P4"
PARAMETER_BITWIDTH_2 = "P2"





class RoundSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Rounding.
    
    During Forward pass: returns round(input).
    During Backward pass: passes gradient through unchanged (identity).
    """
    @staticmethod
    def forward(ctx, input) -> torch.Tensor:
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Straight-through: pass gradient

def round_ste(x: torch.Tensor) -> torch.Tensor:
    return RoundSTE.apply(x)


def get_bitwidth_range(bitwidth: int) -> Tuple[int, int]:
    """Get the min/max range for a given bitwidth
    
    Args:
        bitwidth: Number of bits for quantization (e.g., 8)
        
    Returns:
        Tuple of (qmin, qmax) representing the quantization range
    """
    qmin = -(2 ** (bitwidth - 1))
    qmax = (2 ** (bitwidth - 1)) - 1
    return qmin, qmax

def get_quantize_scale_sy(rmax, bitwidth) -> torch.Tensor:
    _, qmax = get_bitwidth_range(bitwidth)
    scale = (rmax / qmax)
    scale[scale == 0] = 1
    return scale

def get_quantize_scale_zero_point_assy(rmax, rmin, bitwidth) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin, qmax = get_bitwidth_range(bitwidth)
    scale = (rmax - rmin) / (qmax - qmin)
    scale_zero_index = (scale == 0)
    scale[scale_zero_index] = 1

    # zero_point = torch.round(qmin - (rmin / scale))
    zero_point = round_ste(qmin - (rmin / scale))
    zero_point[scale_zero_index] = 0
    return scale, zero_point


def get_quantize_scale_per_tensor_sy(tensor_real: torch.Tensor, 
                                   bitwidth: int = 8, 
                                   metric: str = "l2") -> torch.Tensor:
    """Calculate symmetric per-tensor quantization scale
    
    Args:
        tensor_real: Input tensor to quantize
        bitwidth: Number of bits for quantization
        metric: Scaling metric (currently only "l2" supported)
        
    Returns:
        Calculated scale factor
    """
    _, qmax = get_bitwidth_range(bitwidth)
    rmax = tensor_real.abs().max()
    scale = (rmax / qmax)
    scale[scale == 0] = 1
    return scale

def get_quantize_scale_zero_point_per_tensor_assy(tensor_real: torch.Tensor, 
                                                bitwidth: int = 8, 
                                                metric: str = "l2") -> tuple:
    """Calculate asymmetric per-tensor quantization scale and zero point
    
    Args:
        tensor_real: Input tensor to quantize
        bitwidth: Number of bits for quantization
        metric: Scaling metric (currently only "l2" supported)
        
    Returns:
        Tuple of (scale, zero_point)
    """
    qmin, qmax = get_bitwidth_range(bitwidth)
    rmin = tensor_real.min()
    rmax = tensor_real.max()
    
    scale = (rmax - rmin) / (qmax - qmin)
    scale_zero_index = (scale == 0)
    scale[scale_zero_index] = 1

    # zero_point = torch.round(qmin - (rmin / scale))
    zero_point = round_ste(qmin - (rmin / scale))
    zero_point[scale_zero_index] = 0
    return scale, zero_point


def get_quantize_scale_per_channel_sy(tensor_real: torch.Tensor, 
                                    bitwidth: int = 8, 
                                    metric: str = "l2") -> torch.Tensor:
    """Calculate symmetric per-channel quantization scale
    
    Args:
        tensor_real: Input tensor to quantize
        bitwidth: Number of bits for quantization
        metric: Scaling metric (currently only "l2" supported)
        
    Returns:
        Calculated scale factors per channel
    """
    _, qmax = get_bitwidth_range(bitwidth)
    rmax = tensor_real.abs().view(tensor_real.size(0), -1).max(dim=1)[0]
    scale = (rmax / qmax)
    scale[scale == 0] = 1
    return scale

def get_quantize_scale_zero_point_per_channel_assy(tensor_real: torch.Tensor, 
                                               bitwidth: int = 8, 
                                               metric: str = "l2") -> tuple:
    """Calculate asymmetric per-channel quantization scale and zero point
    
    Args:
        tensor_real: Input tensor to quantize
        bitwidth: Number of bits for quantization
        metric: Scaling metric (currently only "l2" supported)
        
    Returns:
        Tuple of (scale, zero_point) per channel
    """
    qmin, qmax = get_bitwidth_range(bitwidth)
    rmin = tensor_real.view(tensor_real.size(0), -1).min(dim=1)[0]
    rmax = tensor_real.view(tensor_real.size(0), -1).max(dim=1)[0]
    
    scale = (rmax - rmin) / (qmax - qmin)
    scale_zero_index = (scale == 0)
    scale[scale_zero_index] = 1
    
    zero_point = round_ste(qmin - (rmin / scale))
    # zero_point = torch.round(qmin - (rmin / scale))
    zero_point[scale_zero_index] = 0
    
    return scale, zero_point

def quantize_per_tensor_sy(tensor_real: torch.Tensor, 
                          scale: torch.Tensor, 
                          bitwidth: int = 8, 
                          dtype=None) -> torch.Tensor:
    """Symmetric per-tensor quantization
    
    Args:
        tensor_real: Input tensor to quantize
        scale: Quantization scale factor
        bitwidth: Number of bits for quantization
        dtype: Output data type
        
    Returns:
        Quantized tensor
    """
    _, qmax = get_bitwidth_range(bitwidth)

    if dtype is None:
        return torch.clamp(
            round_ste(tensor_real / scale), -qmax, qmax
            # torch.round(tensor_real / scale), -qmax, qmax
        )
    return torch.clamp(
        round_ste(tensor_real / scale), -qmax, qmax
    ).to(dtype)

def dequantize_per_tensor_sy(tensor_quant: torch.Tensor, 
                            scale: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor dequantization
    
    Args:
        tensor_quant: Quantized input tensor
        scale: Quantization scale factor
        
    Returns:
        Dequantized tensor
    """
    return tensor_quant * scale

def fake_quantize_per_tensor_sy(
    tensor_real: torch.Tensor, 
    scale: torch.Tensor, 
    bitwidth: int = 8
): 
    tensor_quant = quantize_per_tensor_sy(tensor_real, scale, bitwidth)
    return dequantize_per_tensor_sy(tensor_quant, scale)
    

def quantize_per_tensor_assy(tensor_real: torch.Tensor, 
                            scale: torch.Tensor, 
                            zero_point: torch.Tensor, 
                            bitwidth: int = 8, 
                            dtype=None) -> torch.Tensor:
    """Asymmetric per-tensor quantization
    
    Args:
        tensor_real: Input tensor to quantize
        scale: Quantization scale factor
        zero_point: Quantization zero point
        bitwidth: Number of bits for quantization
        dtype: Output data type
        
    Returns:
        Quantized tensor
    """
    qmin, qmax = get_bitwidth_range(bitwidth)
    
    if dtype is None:
        return torch.clamp(
            # torch.round(tensor_real / scale) + zero_point, qmin, qmax
            round_ste(tensor_real / scale) + zero_point, qmin, qmax
        )
    
    return torch.clamp(
        # torch.round(tensor_real / scale) + zero_point, qmin, qmax
        round_ste(tensor_real / scale) + zero_point, qmin, qmax
    ).to(dtype)
    
    

def dequantize_per_tensor_assy(tensor_quant: torch.Tensor, 
                              scale: torch.Tensor, 
                              zero_point: torch.Tensor) -> torch.Tensor:
    """Asymmetric per-tensor dequantization
    
    Args:
        tensor_quant: Quantized input tensor
        scale: Quantization scale factor
        zero_point: Quantization zero point
        
    Returns:
        Dequantized tensor
    """
    return (tensor_quant - zero_point) * scale



def fake_quantize_per_tensor_assy(
    tensor_real: torch.Tensor, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor, 
    bitwidth: int = 8
): 
    tensor_quant = quantize_per_tensor_assy(tensor_real, scale, zero_point, bitwidth)
    return dequantize_per_tensor_assy(tensor_quant, scale, zero_point)
    

def quantize_per_channel_sy(tensor_real: torch.Tensor, 
                           scale: torch.Tensor, 
                           bitwidth: int = 8, 
                           dtype=None) -> torch.Tensor:
    """Symmetric per-channel quantization
    
    Args:
        tensor_real: Input tensor to quantize
        scale: Quantization scale factors per channel
        bitwidth: Number of bits for quantization
        dtype: Output data type
        
    Returns:
        Quantized tensor
    """
    _, qmax = get_bitwidth_range(bitwidth)
    broadcast_shape = [1] * tensor_real.ndim
    broadcast_shape[0] = -1
    
    if dtype is None:
        return torch.clamp(
            # torch.round(tensor_real / scale.view(*broadcast_shape)), -qmax, qmax
            round_ste(tensor_real / scale.view(*broadcast_shape)), -qmax, qmax
        )
    return torch.clamp(
        round_ste(tensor_real / scale.view(*broadcast_shape)), -qmax, qmax
    ).to(dtype)
    
def dequantize_per_channel_sy(tensor_quant: torch.Tensor, 
                             scale: torch.Tensor) -> torch.Tensor:
    """Symmetric per-channel dequantization
    
    Args:
        tensor_quant: Quantized input tensor
        scale: Quantization scale factors per channel
        
    Returns:
        Dequantized tensor
    """
    broadcast_shape = [1] * tensor_quant.ndim
    broadcast_shape[0] = -1
    return tensor_quant * scale.view(*broadcast_shape)

def fake_quantize_per_channel_sy(tensor_real: torch.Tensor, 
                           scale: torch.Tensor, 
                           bitwidth: int = 8, 
                           ) -> torch.Tensor:
    torch_quant = quantize_per_channel_sy(tensor_real, scale, bitwidth)
    return dequantize_per_channel_sy(torch_quant, scale)

def quantize_per_channel_assy(tensor_real: torch.Tensor, 
                             scale: torch.Tensor, 
                             zero_point: torch.Tensor, 
                             bitwidth: int = 8, 
                             dtype=None) -> torch.Tensor:
    """Asymmetric per-channel quantization
    
    Args:
        tensor_real: Input tensor to quantize
        scale: Quantization scale factors per channel
        zero_point: Quantization zero points per channel
        bitwidth: Number of bits for quantization
        dtype: Output data type
        
    Returns:
        Quantized tensor
    """
    qmin, qmax = get_bitwidth_range(bitwidth)
    broadcast_shape = [1] * tensor_real.ndim
    broadcast_shape[0] = -1
    
    if dtype is None:
        return torch.clamp(
            # torch.round(tensor_real / scale.view(*broadcast_shape)) + zero_point.view(*broadcast_shape), 
            round_ste(tensor_real / scale.view(*broadcast_shape)) + zero_point.view(*broadcast_shape), 
            qmin, qmax
        )
    return torch.clamp(
        # torch.round(tensor_real / scale.view(*broadcast_shape)) + zero_point.view(*broadcast_shape), 
        round_ste(tensor_real / scale.view(*broadcast_shape)) + zero_point.view(*broadcast_shape), 
        qmin, qmax
    ).to(dtype)
    

def dequantize_per_channel_assy(tensor_quant: torch.Tensor, 
                               scale: torch.Tensor, 
                               zero_point: torch.Tensor) -> torch.Tensor:
    """Asymmetric per-channel dequantization
    
    Args:
        tensor_quant: Quantized input tensor
        scale: Quantization scale factors per channel
        zero_point: Quantization zero points per channel
        
    Returns:
        Dequantized tensor
    """
    broadcast_shape = [1] * tensor_quant.ndim
    broadcast_shape[0] = -1
    return (tensor_quant - zero_point.view(*broadcast_shape)) * scale.view(*broadcast_shape)


def fake_quantize_per_channel_assy(tensor_real: torch.Tensor, 
                             scale: torch.Tensor, 
                             zero_point: torch.Tensor, 
                             bitwidth: int = 8, 
                             ) -> torch.Tensor:
    tensor_quant = quantize_per_channel_assy(tensor_real, scale, zero_point, bitwidth)
    return dequantize_per_channel_assy(tensor_quant, scale, zero_point)


def float32_to_bytes(val: float) -> list:
    """Convert float32 value to bytes (little-endian)
    
    Args:
        val: Float value to convert
        
    Returns:
        List of bytes
    """
    return list(struct.pack("<f", val))

def int32_to_bytes(val: int) -> list:
    """Convert int32 value to bytes (little-endian)
    
    Args:
        val: Integer value to convert
        
    Returns:
        List of bytes
    """
    return list(struct.pack("<i", val))

def int8_to_bytes(val: int) -> list:
    """Convert int8 value to bytes
    
    Args:
        val: Integer value to convert
        
    Returns:
        List of bytes
    """
    return list(struct.pack("<b", val))

def int4_to_bytes(data: list) -> list:
    """Pack list of 4-bit values into bytes
    
    Args:
        data: List of 4-bit values to pack
        
    Returns:
        List of bytes
    """
    byte = 0
    for val in data[::-1]:
        byte = ((byte << 4) | (val & 0x0F))
    return list(struct.pack("<b", byte))


def int2_to_bytes(data: list) -> list:
    """Pack list of 2-bit values into bytes
    
    Args:
        data: List of 4-bit values to pack
        
    Returns:
        List of bytes
    """
    byte = 0
    for val in data[::-1]:
        byte = ((byte << 2) | (val & 0x03))
    return list(struct.pack("<b", byte))

def pack_into_byte(byte_list, bitwidth):
    """Packs a list of bytes into a single byte

    Args:
        byte_list: List of containing data to be packed
        bitwidth: the bitwidth which the data are, i.e. 8, 4, 2
    """
    assert len(byte_list) <= 8 // bitwidth, f"byte list of lenght {len(byte_list)} cannot be pack into 8 bit withs bitwidth {bitwidth}"
    shift = bitwidth
    mask = (1 << bitwidth) - 1
    byte = 0
    for value in reversed(byte_list):
        byte = ((byte << shift) | (value & mask))
    return list(struct.pack("<b", byte))

def convert_tensor_to_bytes_var(tensor: torch.Tensor, 
                               var_name: str, 
                               bitwidth: Optional[int] = 8,
                               for_arduino = False) -> tuple:
    """Convert tensor to C-style byte array declaration
    
    Args:
        tensor: Input tensor to convert
        var_name: Name to use for variable
        bitwidth: Bitwidth for quantization (if applicable)
        for_arduino: If True, generates Arduino-compatible C code, add PROGMEM if needed to ensure the params are stored in flash memory.
        
    Returns:
        Tuple of (header string, definition string)
    """
    if tensor.dtype == torch.float:
        byte_convert = float32_to_bytes
        byte_per_line = FLOAT32_BYTE_PER_LINE
    elif tensor.dtype == torch.int32:
        byte_convert = int32_to_bytes
        byte_per_line = INT32_BYTE_PER_LINE
    else:
        byte_convert = int8_to_bytes
        byte_per_line = INT8_BYTE_PER_LINE

    if not for_arduino:
        var_header_str = f"extern const uint8_t {var_name}[];\n"
        var_def_str = f"\nconst uint8_t {var_name}[] = {{\n"
    else:
        # Fix to enforce that the weight are loaded in flash and not in RAM.
        # By default Arduino loads all global variable on the RAM.
        # https://docs.arduino.cc/language-reference/en/variables/utilities/PROGMEM/
        var_header_str = f"extern const uint8_t {var_name}[] PROGMEM;\n"
        var_def_str = f"\nconst uint8_t {var_name}[] PROGMEM = {{\n"

    if tensor.dtype != torch.int8 or bitwidth == 8:
        # Standard byte conversion for non-packed data
        for line in torch.split(tensor.flatten(), byte_per_line):
            var_def_str += "    " + ", ".join(
                [f"0x{b:02X}" for val in line.tolist() for b in byte_convert(val)]
            ) + ",\n"
    else:
        # Special handling for packed data (4-bit, 2-bit, etc.)
        if bitwidth is not None:
            data_per_byte = 8 // bitwidth
            # for i in range(math.ceil(len(line)/data_per_byte)):
        tensor = tensor.flatten()

        for line in torch.split(tensor.flatten(), INT8_BYTE_PER_LINE * data_per_byte):
            bytes = []
            for i in range(math.ceil(len(line)/data_per_byte)):
                data = []
                for pos in range(data_per_byte):
                    index = (i*data_per_byte)+pos
                    if index < len(line):
                        data.append(line[index])
                bytes.append(pack_into_byte(data, bitwidth))

            var_def_str += "    " + ", ".join(
                [f"0x{b:02X}" for val in bytes for b in val]
            ) + ",\n"
            
    var_def_str += "};\n"
    return var_header_str, var_def_str



def get_size_in_bits(var: Any, is_packed:bool = False, bitwidth:int = 8) -> int:
    """Returns the size of a variable in bits."""

    # Handle ints and floats as scalars
    if isinstance(var, int) or isinstance(var, float):
        return 32  # assuming 32-bit for both

    # Handle nn.Parameter
    if isinstance(var, torch.nn.Parameter):
        var = var.data  # extract the underlying tensor

    # Handle torch.Tensor
    if isinstance(var, torch.Tensor):
        if var.dtype == torch.int8:
            dtype_size = 8
        elif var.dtype == torch.int32:
            dtype_size = 32
        elif var.dtype == torch.float32:
            dtype_size = 32
        elif var.dtype == torch.float64:
            dtype_size = 64
        else:
            raise RuntimeError(f"get_size for dtype {var.dtype} not implemented!")
        
        numel = var.numel()

        if var.dtype != torch.int8:
            return numel * dtype_size
            
        if is_packed:
            data_per_byte = 8 // bitwidth
            numel = math.ceil(numel/data_per_byte)

        return numel * dtype_size

    raise RuntimeError(f"get_size for type {type(var)} not implemented!")
