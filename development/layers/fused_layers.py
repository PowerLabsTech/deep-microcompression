"""
@file fused_layers.py
@brief Composite Layers for Optimized Inference.

Role in DMC Pipeline:
1. Memory Efficiency: performing the activation immediately after the MAC operation
   avoids storing the intermediate pre-activation tensor to the SRAM workspace.
2. C-Code Generation: generates unified C loops (e.g. acc = max(0, acc)) rather
   than separate function calls.

Quantization contract: the input arrives already fake-quantized by the previous
layer's output_quantize. These fused layers do NOT re-apply input_quantize —
that would double-quantize the input. The input_quantize observer is only used
for calibration (tracking the input distribution), not for re-quantizing at
forward time.
"""

import torch
from torch import nn

from .layer import Layer
from .linear import Linear
from .conv import Conv2d
from .activation import ReLU, ReLU6


class LinearReLU(Linear):
    """Fused Linear + ReLU Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)
            if self.is_quantized:
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        output = self.relu.forward(nn.functional.linear(input, weight, bias))

        if self.is_compressed and self.is_quantized and hasattr(self, "output_quantize"):
            output = self.output_quantize(output)
        return output


class LinearReLU6(Linear):
    """Fused Linear + ReLU6 Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu6 = ReLU6()

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)
            if self.is_quantized:
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        output = self.relu6.forward(nn.functional.linear(input, weight, bias))

        if self.is_compressed and self.is_quantized and hasattr(self, "output_quantize"):
            output = self.output_quantize(output)
        return output


class Conv2dReLU(Conv2d):
    """Fused Conv2d + ReLU Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = ReLU()

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)
            if self.is_quantized:
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        output = self.relu.forward(nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        ))

        if self.is_compressed and self.is_quantized and hasattr(self, "output_quantize"):
            output = self.output_quantize(output)
        return output


class Conv2dReLU6(Conv2d):
    """Fused Conv2d + ReLU6 Layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu6 = ReLU6()

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if self.is_compressed:
            if self.is_pruned_channel:
                weight = self.weight_prune_channel(weight)
                if self.bias is not None:
                    bias = self.bias_prune_channel(bias)
            if self.is_quantized:
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        output = self.relu6.forward(nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        ))

        if self.is_compressed and self.is_quantized and hasattr(self, "output_quantize"):
            output = self.output_quantize(output)
        return output
