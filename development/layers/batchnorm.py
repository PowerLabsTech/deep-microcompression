from typing import Optional

import torch
from torch import nn

from .layer import Layer
from ..compressors import Prune_Channel

from ..utils import (
    convert_tensor_to_bytes_var,
    get_size_in_bits,

    QuantizationScheme
)

class BatchNorm2d(Layer, nn.BatchNorm2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def forward(self, input):
        
        return super().forward(input)
    
    @property
    def folded_weight(self):
        return self.weight / torch.sqrt(self.running_var + self.eps)
    

    @property
    def folded_bias(self):
        return self.bias - self.running_mean * self.weight / torch.sqrt(self.running_var + self.eps)
   
    @torch.no_grad
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        setattr(self, "prune_channel", Prune_Channel(
            module=self, keep_current_channel_index=keep_prev_channel_index
        ))
        return keep_prev_channel_index
    

    def get_prune_channel_possible_hypermeters(self):
        return None


    @torch.no_grad()
    def init_quantize(self, bitwidth, scheme, granularity, previous_output_quantize = None):

        # if scheme == QuantizationScheme.STATIC:
        #     raise RuntimeError("Can not perform static quantization with BatchNorm2d, fuse the model first!")
            
        return previous_output_quantize


    def get_size_in_bits(self):
        
        folded_weight, folded_bias = self.get_compression_parameters()
        size = 0

        size += get_size_in_bits(folded_weight)
        size += get_size_in_bits(folded_bias)

        return size

    def get_compression_parameters(self):

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
    def convert_to_c(self, var_name, input_shape):

        folded_weight, folded_bias = self.get_compression_parameters()

        input_row_size, input_col_size = input_shape[1:]

        input_channel_size = folded_weight.size(0)

        param_header, param_def = convert_tensor_to_bytes_var(folded_weight, f"{var_name}_folded_weight")
        layer_header = param_header
        layer_param_def = param_def

        param_header, param_def = convert_tensor_to_bytes_var(folded_bias, f"{var_name}_folded_bias")
        layer_header += param_header
        layer_param_def += param_def

        layer_def = (
            f"{self.__class__.__name__} {var_name}({input_channel_size}, {input_row_size}, {input_col_size}, "
            f"(float*){var_name}_folded_weight, (float*){var_name}_folded_bias);\n"
        )

        layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        
        return layer_header, layer_def, layer_param_def