"""
@file flatten.py
@brief Flatten Layer with Structured Pruning Coordination.

    In the DMC pipeline, the Flatten layer is functionally simple (reshaping tensors)
    but architecturally critical for Structured Pruning (Section III-A).

    Role in Pruning:
    When a Convolutional channel is pruned, an entire 2D feature map (H x W)
    disappears. The Flatten layer must translate the "Surviving Channel Indices"
    into a list of "Surviving Flat Element Indices" so the subsequent Linear layer
    knows which input connections to remove.
"""

import math
from typing import Optional

import torch
from torch import nn

from .layer import Layer
from ..compressors import Quantize, QuantizationScheme, QuantizationBitWidthError

from ..utils import (
    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2,
)


class Flatten(Layer, nn.Flatten):
    """
    DMC-aware Flatten layer.

    Responsibilities:
    1. Shape Transformation: Standard NCHW -> N(C*H*W) flattening.
    2. Pruning Translation: Converts keep_channel_mask (spatial) to
       keep_neuron_mask (flat) for dense matrix connectivity.
    3. Quantization Pass-through: The base Layer.init_quantize handles scale/zero-point
       forwarding unchanged — Flatten does not override it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        return super().forward(input)

    @torch.no_grad()
    def init_prune_channel(
        self,
        sparsity: float,
        input_shape: torch.Size,
        keep_prev_channel_index: Optional[torch.Tensor],
        keep_current_channel_index: Optional[torch.Tensor],
        is_output_layer: bool = False,
        metric: str = "l2",
    ):
        """Translates per-channel keep-indices into per-element keep-indices for Linear."""
        channel_numel = input_shape[1:].numel()

        if keep_current_channel_index is not None:
            return keep_current_channel_index

        # Expand surviving channel indices to cover all spatial elements per channel
        if keep_prev_channel_index is None:
            keep_prev_channel_index = torch.arange(input_shape[0])

        start_positions = keep_prev_channel_index * channel_numel
        channel_elements_index = torch.arange(channel_numel).to(start_positions.device)
        keep_current_channel_index = start_positions.view(-1, 1) + channel_elements_index
        return keep_current_channel_index.flatten()

    def get_size_in_bits(self):
        return 0

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return (math.ceil(input_shape.numel() / data_per_byte)
                + math.ceil(self.get_output_tensor_shape(input_shape).numel() / data_per_byte))

    def get_output_tensor_shape(self, input_shape: torch.Size):
        return torch.Size((input_shape.numel(),))

    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        input_size = torch.Size(input_shape).numel()
        scheme = None
        if self.is_quantized and "quantize" in self.__dict__["_dmc"]:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]

        if scheme != QuantizationScheme.STATIC:
            layer_def = f"{self.__class__.__name__} {var_name}({input_size});\n"
            layer_header = f"extern {self.__class__.__name__} {var_name};\n\n"
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

            layer_def = f"{self.__class__.__name__}_SQ {var_name}({input_size}, {quantize_property});\n"
            layer_header = f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

        return layer_header, layer_def, ""
