"""
@file fused_layers.py
@brief Composite Layers for Optimized Inference.

This module implements the fused layer structures (Conv+ReLU, Linear+ReLU) 
created by `fuse.py`.

Role in DMC Pipeline:
1.  Memory Efficiency: By performing the activation function 
    immediately after the MAC (Multiply-Accumulate) operation—while the value 
    is still in the CPU register/accumulator—we avoid storing the intermediate 
    pre-activation tensor to the SRAM workspace. This is critical for fitting 
    models like LeNet-5 into the 2KB RAM of the ATmega32.
    
2.  C-Code Generation: These classes allow the `convert_to_c` 
    engine to generate unified C loops (e.g., `acc = max(0, acc)`) rather than 
    separate function calls.
"""

import torch
from torch import nn

from .layer import Layer
from .linear import Linear
from .conv import Conv2d
from .activation import ReLU, ReLU6


class LinearReLU(Linear):

    def __init__(self, *args, **kwargs):
        """Fused Linear + ReLU Layer"""
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
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        input = nn.functional.linear(input, weight, bias)
        output = self.relu.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
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
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)

        input = nn.functional.linear(input, weight, bias)
        output = self.relu6.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
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
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)
                    
        input = nn.functional.conv2d(
            input, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        output = self.relu.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
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
                if hasattr(self, "input_quantize"): 
                    input = self.input_quantize(input)
                weight = self.weight_quantize(weight)
                if self.bias is not None and hasattr(self, "bias_quantize"):
                    bias = self.bias_quantize(bias)
                            
        input = nn.functional.conv2d(
            input, weight, bias,
            self.stride, self.padding,
            self.dilation, self.groups
        )
        output = self.relu6.forward(input)

        if self.is_compressed:
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)
        return output
    


    
