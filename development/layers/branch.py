import math
from typing import Optional
import warnings

import torch
from torch import nn

from .layer import Layer

from .activation import ReLU, ReLU6
from .batchnorm import BatchNorm2d
from .flatten import Flatten
from .padding import ConstantPad2d
from .pooling import AvgPool2d, MaxPool2d

from ..compressors import (
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
)

from ..utils import (
    get_size_in_bits,

    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2
)

class Branch(Layer, nn.Module):

    NON_OUTPUT_MODIFYING_LAYERS = (
        AvgPool2d,
        BatchNorm2d,
        ConstantPad2d,
        Flatten,
        ReLU,
        ReLU6,
        MaxPool2d,
    )
    def __init__(self, sublayer1:Layer, sublayer2:Optional[Layer]=None):

        if (sublayer1 not in self.NON_OUTPUT_MODIFYING_LAYERS and (sublayer2 in self.NON_OUTPUT_MODIFYING_LAYERS or sublayer2 is None)) or \
            sublayer2 not in self.NON_OUTPUT_MODIFYING_LAYERS and sublayer1 in self.NON_OUTPUT_MODIFYING_LAYERS:
            warnings.warn((f"sublayer 1 of type {type(sublayer1)} and sublayer2 {sublayer2} have only one of them"
                           "as a compression parameter modifying layer (it recomputes its parameters) like Linear or Conv"
                            " but the other uses the parameters from the previous layers, this will result in the modifying"
                            f" layer having it parameters tied to that of the non modifying layer."))
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
            # Rescale the sum of 2 intb back to inb
            # TODO: Add integer division loss, maybe quantize it back to int before dividing
            if self.is_quantized:
                output /= 2
            #     if hasattr(self, "output_quantize"):
            #         output = self.output_quantize(output)
        return output
    

    def init_prune_channel(
        self, 
        sparsity: float, 
        input_shape: torch.Size,
        keep_prev_channel_index:Optional[torch.Tensor], 
        keep_current_channel_index:Optional[torch.Tensor],
        is_output_layer: bool = False, 
        metric: str = "l2"
    )-> Optional[torch.Tensor]:
        # TODO: Figure out how to solve the channel mixup problem with skip connections when prunning,
        #       when a layer like linear is used the neurons positions get mixed up and if it is pruned
        #       the neurons of the left over neurons don't align

        # TODO: Figure out how to make branch layer the last layer, during pruning the input will have 
        #       less number of neurons, if it is the last layer, which the output shape has to be maintained
        #       there will be a shape mismatch
        if is_output_layer:
            raise NotImplementedError("Branch layer has not been implemented for being the last layer.")
        
        # NOTE: A temporary fix to the pruning mix up problem, when a non mixing layer is one of the layers
        #       make sure the other can not choose which channel or filter to prune out
        if isinstance(self.sublayer2, self.NON_OUTPUT_MODIFYING_LAYERS) or self.sublayer2 is None:
            keep_current_channel_index = keep_prev_channel_index

        keep_prev_channel_index1 = self.sublayer1.init_prune_channel(
            sparsity, input_shape, keep_prev_channel_index, keep_current_channel_index=keep_current_channel_index,
            is_output_layer=is_output_layer, metric=metric
        )
        input_shape1 = self.sublayer1.get_output_tensor_shape(torch.Size(input_shape))

        if self.sublayer2 is not None:
            keep_prev_channel_index2 = self.sublayer2.init_prune_channel(
                sparsity, input_shape, keep_prev_channel_index, keep_current_channel_index=keep_prev_channel_index1,
                is_output_layer=is_output_layer, metric=metric, 
            )
            assert torch.equal(keep_prev_channel_index1.cpu(), keep_prev_channel_index2.cpu()), (
                f"The keep_prev_channel_index of submodule1 {keep_prev_channel_index1}"
                f" and submodule2 {keep_prev_channel_index2} aren't the same."
            )
            input_shape2 = self.sublayer2.get_output_tensor_shape(torch.Size(input_shape))
        else:
            input_shape2 = input_shape

        assert input_shape1 == input_shape2, (
            f"The output shape of output of submodule1 ({self.sublayer1}) {input_shape1}"
            f" and submodule2 ({self.sublayer2}) {input_shape2} aren't the same after pruning."
        )

        return keep_prev_channel_index1


    def init_quantize(
        self, 
        parameter_bitwidth: int, 
        granularity: QuantizationGranularity, 
        scheme: QuantizationScheme,
        activation_bitwidth:Optional[int]=None,
        previous_output_quantize: Optional[Quantize] = None,
        current_output_quantize: Optional[Quantize] = None,
    ):
        super().init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)

        if scheme != QuantizationScheme.STATIC:
            self.sublayer1.init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)
            if self.sublayer2 is not None:
                self.sublayer2.init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)
            return

        assert previous_output_quantize is not None, "Pass a quantizer for the input, it is usually from the preceeding layer."
        assert activation_bitwidth is not None, "Pass in a activation bitwidth when do static quantization"

        # To force both layers to have the same quantization scale and zeropoint to make the addition simpler
        if isinstance(self.sublayer2, self.NON_OUTPUT_MODIFYING_LAYERS) or self.sublayer2 is None:
            current_output_quantize = previous_output_quantize

        next_output_quantize1 = self.sublayer1.init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize, current_output_quantize)
        if self.sublayer2:
            next_output_quantize2 = self.sublayer2.init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize, next_output_quantize1)
            assert len(next_output_quantize2.base) == 1 and (
                next_output_quantize1 == next_output_quantize2.base[0] or 
                next_output_quantize1.base[0] == next_output_quantize2.base[0]
                ), (
                f"The next_output_quantize of submodule1 {next_output_quantize1}"
                f" and the quantization base of submodule2 {next_output_quantize2.base} aren't the same."
            )
        return next_output_quantize1


    
    def get_prune_channel_possible_hyperparameters(self):
        return self.sublayer1.get_prune_channel_possible_hyperparameters()

    def get_quantize_possible_hyperparameters(self):
        # TODO: Fix this to return the possible hyperparameters for both sublayers
        return self.sublayer1.get_quantize_possible_hyperparameters()


    def get_compression_parameters(self):
        return


    def get_size_in_bits(self) -> int:

        size = self.sublayer1.get_size_in_bits()
        if self.sublayer2 is not None:
            size += self.sublayer2.get_size_in_bits()
        if self.is_compressed:
            # Rescalling
            if self.is_quantized:
                if hasattr(self, "output_quantize"):
                    size += get_size_in_bits(self.output_quantize.scale)
                    size += get_size_in_bits(self.output_quantize.zero_point)
        return size


    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        workspace_size = self.sublayer1.get_workspace_size(input_shape, data_per_byte)
        if self.sublayer2 is not None:
            workspace_size += math.ceil(self.sublayer2.get_output_tensor_shape(input_shape).numel() / data_per_byte)
        return workspace_size


    def get_output_tensor_shape(self, input_shape):
        return self.sublayer1.get_output_tensor_shape(input_shape)


    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        input_shape = torch.Size(input_shape)
        layer_header, layer_def, layer_param_def = self.sublayer1.convert_to_c(f"{var_name}_sublayer1", input_shape, for_arduino=for_arduino)
        if self.sublayer2 is not None:
            layer_header2, layer_def2, layer_param_def2 = self.sublayer2.convert_to_c(f"{var_name}_sublayer2", input_shape, for_arduino=for_arduino)

            layer_header += layer_header2
            layer_def += layer_def2
            layer_param_def += layer_param_def2

        scheme = None
        if self.is_quantized:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
            activation_bitwidth = self.__dict__["_dmc"]["quantize"]["activation_bitwidth"]

        if scheme != QuantizationScheme.STATIC:
            if self.sublayer2 is not None:
                layer_def += f"{self.__class__.__name__} {var_name}(&{var_name}_sublayer1, &{var_name}_sublayer2);\n"
            else:
                layer_def += f"{self.__class__.__name__} {var_name}(&{var_name}_sublayer1, nullptr);\n"
            layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        else:

            quantize_property = ""

            if activation_bitwidth == 8:
                quantize_property += ACTIVATION_BITWIDTH_8
            elif activation_bitwidth == 4:
                quantize_property += ACTIVATION_BITWIDTH_4
            elif activation_bitwidth == 2:
                quantize_property += ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError
            
            if self.sublayer2 is not None:
                layer_def += f"{self.__class__.__name__}_SQ {var_name}(&{var_name}_sublayer1, &{var_name}_sublayer2, {quantize_property});\n"
            else:
                layer_def += f"{self.__class__.__name__}_SQ {var_name}(&{var_name}_sublayer1, nullptr, {quantize_property});\n"
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"

        
        return layer_header, layer_def, layer_param_def
    



