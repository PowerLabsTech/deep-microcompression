from typing import Optional

import torch
from torch import nn

from .layer import Layer
from ..compressors import (
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
)

from ..utils import (
    convert_tensor_to_bytes_var,
    get_size_in_bits,

    STATIC_BIAS_BITWDHT,
)

class BranchLayer(Layer, nn.Module):

    def __init__(self, sublayer1:Layer, sublayer2:Optional[Layer]=None):

        super().__init__()
        self.sublayer1 = sublayer1
        self.sublayer2 = sublayer2

    def forward(self, input):

        output1 = self.sublayer1(input)

        
        if self.sublayer2 is not None:
            output2 = self.sublayer2(input)
        else:
            output2 = input

        assert output1.size() == output2.size(), (
            f"The output shape of output of submodule1 {output1.size()}"
            f" and submodule2 {output2.size()} aren't the same."
        )

        output = output1 + output2

        if self.is_compressed:
            # Rescalling
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    output = self.output_quantize(output)

        return output
    

    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ) -> Optional[torch.Tensor]:
        
        
        keep_prev_channel_index1 = self.sublayer1.init_prune_channel(
            sparsity, keep_prev_channel_index, input_shape,
            is_output_layer=is_output_layer, metric=metric
        )
        _, input_shape1 = self.sublayer1.get_output_tensor_shape(torch.Size(input_shape))

        if self.sublayer2 is not None:
            keep_prev_channel_index2 = self.sublayer2.init_prune_channel(
                sparsity, keep_prev_channel_index, input_shape,
                is_output_layer=is_output_layer, metric=metric, keep_current_channel_index=keep_prev_channel_index1
            )
            _, input_shape2 = self.sublayer2.get_output_tensor_shape(torch.Size(input_shape))

            assert input_shape1 == input_shape2, (
                f"The output shape of output of submodule1 {input_shape1}"
                f" and submodule2 {input_shape2} aren't the same."
            )

            assert keep_prev_channel_index1 == keep_prev_channel_index2, (
                f"The keep_prev_channel_index of submodule1 {keep_prev_channel_index1}"
                f" and submodule2 {keep_prev_channel_index2} aren't the same."
            )

        return keep_prev_channel_index1




    def init_quantize(
        self, 
        bitwidth: int, 
        scheme: QuantizationScheme, 
        granularity: QuantizationGranularity, 
        previous_output_quantize: Optional[Quantize] = None
    ):

        if scheme != QuantizationScheme.STATIC:
            self.sublayer1.init_quantize(bitwidth, scheme, granularity)
            if self.sublayer2 is not None:
                self.sublayer2.init_quantize(bitwidth, scheme, granularity)
            return
    

        assert previous_output_quantize is not None, "Pass a quantizer for the input, it is usually from the preceeding layer."

        setattr(self, "output_quantize", Quantize(
            self, bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
        ))
        
        next_output_quantize1 = self.sublayer1.init_quantize(bitwidth, scheme, granularity, previous_output_quantize)
        if self.sublayer2:
            next_output_quantize2 = self.sublayer1.init_quantize(bitwidth, scheme, granularity, previous_output_quantize)
            assert next_output_quantize1 == next_output_quantize2, (
                f"The next_output_quantize of submodule1 {next_output_quantize1}"
                f" and submodule2 {next_output_quantize2} aren't the same."
            )
        
        if hasattr(self, "output_quantize"):
            return self.output_quantize 
        return None


    
    def get_prune_channel_possible_hyperparameters(self):
        return self.sublayer1.get_prune_channel_possible_hyperparameters()


    def get_compression_parameters(self):
        return


    def get_size_in_bits(self) -> int:

        size = 0
        if self.is_compressed:
            # Rescalling
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    size += get_size_in_bits(self.output_quantize.scale)
                    size += get_size_in_bits(self.output_quantize.zero_point)
        return size

    def get_output_tensor_shape(self, input_shape):
        return self.sublayer1.get_output_tensor_shape(input_shape)

    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        pass



