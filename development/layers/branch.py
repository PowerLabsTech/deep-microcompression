import math
from typing import Optional
import warnings

import torch
from torch import nn

from .layer import Layer

from .activation import ReLU, ReLU6
from .batchnorm import BatchNorm2d
from .block import Block
from .flatten import Flatten
from .padding import ConstantPad2d
from .pooling import AvgPool2d, MaxPool2d

from ..compressors import (
    get_data_bits,
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
    QuantizationBitWidthError,
)

from ..utils import (
    get_size_in_bits,
    pad_bits_to_byte,

    ACTIVATION_BITWIDTH_8,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_2,

    UINT8_T,
    UINT32_T,
    FLOAT_T,
    INT8_T,
    VOID_PTR,
)

class Branch(Layer, nn.Module):

    # FIXME: This should be non scale modifing layer, another list should be
    #        create for non output shape modifying layer
    NON_OUTPUT_MODIFYING_LAYERS = (
        AvgPool2d,
        BatchNorm2d,
        ConstantPad2d,
        Flatten,
        ReLU,
        ReLU6,
        MaxPool2d,
    )
    def __init__(self, sublayer1:Layer, sublayer2:Optional[Layer]=None, input_shape:Optional[torch.Size] = None):

        if (sublayer1 not in self.NON_OUTPUT_MODIFYING_LAYERS and (sublayer2 in self.NON_OUTPUT_MODIFYING_LAYERS or sublayer2 is None)) or \
            sublayer2 not in self.NON_OUTPUT_MODIFYING_LAYERS and sublayer1 in self.NON_OUTPUT_MODIFYING_LAYERS:
            warnings.warn((f"sublayer 1 of type {type(sublayer1)} and sublayer2 {sublayer2} have only one of them"
                           "as a compression parameter modifying layer (it recomputes its parameters) like Linear or Conv"
                            " but the other uses the parameters from the previous layers, this will result in the modifying"
                            f" layer having it parameters tied to that of the non modifying layer."))
        super().__init__()

        self.sublayer1 = sublayer1
        self.sublayer2 = sublayer2

        if self.sublayer2 is not None:
            if input_shape is not None:
                sublayer1_output_shape = self.sublayer1.get_output_tensor_shape()
                sublayer2_output_shape = self.sublayer2.get_output_tensor_shape()
                assert sublayer1_output_shape == sublayer2_output_shape, (
                    f"The output shape of output of sublayer1 {self.sublayer1}: {sublayer1_output_shape}"
                    f" and sublayer2 {self.sublayer2}: {sublayer1_output_shape} aren't the same."
                )
        else:
            # FIXME: There is no verification that if is a container type layer, that it doesnot change the output shape
            if not isinstance(self.sublayer1, (Branch, Block)):    
                assert self.sublayer1 in self.NON_OUTPUT_MODIFYING_LAYERS, (
                    f"If sublayer2 is None, sublayer1 must be a layer that changes the input shape, got {self.sublayer2}"
                )
            

    def forward(self, input):

        output1 = self.sublayer1(input)

        if self.sublayer2 is not None:
            output2 = self.sublayer2(input)
        else:
            output2 = input

        assert output1.size() == output2.size(), (
            f"The output shape of submodule1 {output1.size()}"
            f" and submodule2 {output2.size()} aren't the same."
        )

        output = output1 + output2

        if hasattr(self, "output_quantize"):
            output = self.output_quantize(output)

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

        # Identity shortcut: output channel selection must match the input for residual
        # addition to be shape-compatible. The sparsity value has no effect here — kept
        # channels are fully inherited from the previous prunable layer.
        s1 = sparsity.get("sublayer1", 0) if isinstance(sparsity, dict) else sparsity
        s2 = sparsity.get("sublayer2", 0) if isinstance(sparsity, dict) else sparsity

        if isinstance(self.sublayer2, self.NON_OUTPUT_MODIFYING_LAYERS) or self.sublayer2 is None:
            if (isinstance(s1, float) and s1 > 0.0) or (isinstance(s1, int) and s1 > 0):
                warnings.warn(
                    f"Branch with identity shortcut received sparsity={s1}, but channel "
                    "selection is constrained to match the input (residual compatibility). "
                    "The sparsity value is ignored — channel count is inherited from the previous layer.",
                    UserWarning, stacklevel=2,
                )
            keep_current_channel_index = keep_prev_channel_index

        keep_prev_channel_index1 = self.sublayer1.init_prune_channel(
            s1, input_shape, keep_prev_channel_index, keep_current_channel_index=keep_current_channel_index,
            is_output_layer=is_output_layer, metric=metric
        )
        input_shape1 = self.sublayer1.get_output_tensor_shape(torch.Size(input_shape))

        if self.sublayer2 is not None:
            keep_prev_channel_index2 = self.sublayer2.init_prune_channel(
                s2, input_shape, keep_prev_channel_index, keep_current_channel_index=keep_prev_channel_index1,
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

        pb1   = parameter_bitwidth.get("sublayer1", 8) if isinstance(parameter_bitwidth, dict) else parameter_bitwidth
        pb2   = parameter_bitwidth.get("sublayer2", 8) if isinstance(parameter_bitwidth, dict) else parameter_bitwidth
        gran1 = granularity.get("sublayer1", QuantizationGranularity.PER_TENSOR) if isinstance(granularity, dict) else granularity
        gran2 = granularity.get("sublayer2", QuantizationGranularity.PER_TENSOR) if isinstance(granularity, dict) else granularity

        if scheme != QuantizationScheme.STATIC:
            self.sublayer1.init_quantize(pb1, gran1, scheme, activation_bitwidth, previous_output_quantize)
            if self.sublayer2 is not None:
                self.sublayer2.init_quantize(pb2, gran2, scheme, activation_bitwidth, previous_output_quantize)
            return

        assert previous_output_quantize is not None, "Pass a quantizer for the input, it is usually from the preceeding layer."
        assert activation_bitwidth is not None, "Pass in a activation bitwidth when do static quantization"
        assert current_output_quantize is None, (
            "Branch does not support being used as sublayer2 of another Branch (or any context that passes "
            "a current_output_quantize). Branch always calibrates its output scale from sublayer1 independently "
            "and cannot honor a forced output scale from outside without creating an exponential quantizer chain. "
            "If you need nested branches, provide a custom sublayer2 that is not a Branch."
        )

        # Both sublayers calibrate their output scales independently. The C engine implements
        # the full scaled addition formula:
        #   qb = round(M1*(q1-z1) + M2*(q2-z2)) + zb
        # where M1=s1/sb, M2=s2/sb are precomputed at export time. This means neither sublayer
        # needs to be forced to a particular scale — each finds its own optimal range during QAT.

        next_output_quantize1 = self.sublayer1.init_quantize(
            pb1, gran1, scheme, activation_bitwidth,
            previous_output_quantize, current_output_quantize=None
        )

        if self.sublayer2 is not None:
            next_output_quantize2 = self.sublayer2.init_quantize(
                pb2, gran2, scheme, activation_bitwidth,
                previous_output_quantize, current_output_quantize=None
            )
        else:
            # Identity shortcut: the second operand is the raw input, quantized at the
            # preceding layer's scale.
            next_output_quantize2 = previous_output_quantize

        # branch1/branch2_quantize carry s1/z1 and s2/z2 for the C engine.
        # output_quantize calibrates from the actual post-addition distribution.
        setattr(self, "branch1_quantize", next_output_quantize1)
        if self.sublayer2 is not None:
            setattr(self, "branch2_quantize", next_output_quantize2)
        else:
            setattr(self, "branch2_quantize", previous_output_quantize)
        setattr(self, "output_quantize", Quantize(
            self, activation_bitwidth, scheme, QuantizationGranularity.PER_TENSOR,
            scale_type=QuantizationScaleType.ASSYMMETRIC
        ))
        return self.output_quantize

    def get_quantization_output_parameters(self):
        if not self.is_quantized():
            return

        scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
        if scheme != QuantizationScheme.STATIC:
            return
        
        if hasattr(self, "branch1_quantize") and hasattr(self, "output_quantize"):
            s1_so = self.branch1_quantize.scale / self.output_quantize.scale
            s1z1 = self.branch1_quantize.scale * self.branch1_quantize.zero_point

        if hasattr(self, "branch2_quantize") and hasattr(self, "output_quantize"):
            s2_so = self.branch2_quantize.scale / self.output_quantize.scale
            s2z2 = self.branch2_quantize.scale * self.branch2_quantize.zero_point

        z_o = int(self.output_quantize.zero_point - ((s1z1 + s2z2) / self.output_quantize.scale))
        return s1_so, s2_so, round(z_o)


    def get_prune_channel_possible_hyperparameters(self):
        result = {}
        if (hp := self.sublayer1.get_prune_channel_possible_hyperparameters()) is not None:
            result["sublayer1"] = hp
        if self.sublayer2 is not None:
            if (hp := self.sublayer2.get_prune_channel_possible_hyperparameters()) is not None:
                result["sublayer2"] = hp
        return result if result else None


    def get_quantize_possible_hyperparameters(self):
        result = {}
        if (hp := self.sublayer1.get_quantize_possible_hyperparameters()) is not None:
            result["sublayer1"] = hp
        if self.sublayer2 is not None:
            if (hp := self.sublayer2.get_quantize_possible_hyperparameters()) is not None:
                result["sublayer2"] = hp
        return result if result else None


    def get_compression_parameters(self):
        return


    def get_size_in_bits(self) -> int:

        size = self.sublayer1.get_size_in_bits()
        if self.sublayer2 is not None:
            size += self.sublayer2.get_size_in_bits()
        if self.is_compressed and self.is_quantized:
            quantization_parameters = self.get_quantization_output_parameters()

            if quantization_parameters is not None:
                for param in quantization_parameters:
                    size += get_size_in_bits(param)
        return size


    def get_workspace_size(
        self, input_shape, include_locals=False,
        include_runtime=False, ptr_size=2
    ) -> int:
        if isinstance(input_shape, tuple): input_shape = torch.Size(input_shape)
        data_bits = get_data_bits(self)
        input_workspace_size = pad_bits_to_byte(input_shape.numel() * data_bits) * 2
        
        sublayer1_workspace = self.sublayer1.get_workspace_size(
            input_shape, include_locals, include_runtime, ptr_size)
        if self.sublayer2 is not None:
            sublayer2_workspace = self.sublayer2.get_workspace_size(
                input_shape, include_locals, include_runtime, ptr_size
            )
        else:
            sublayer2_workspace = pad_bits_to_byte(input_shape.numel() * data_bits)
        base = sublayer1_workspace + sublayer2_workspace
        if not (include_locals or include_runtime):
            return base
        scheme = None
        if self.is_quantized:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]
        if scheme == QuantizationScheme.STATIC:
            # uint32_t sublayer1_workspace_size + uint8_t quantize_property
            locals_size  = 5
            # 3 Flash ptrs + 1 workspace alias ptr + 3 fn ptrs + uint32_t i + f32 s1_so + f32 s2_so + i8 zo
            runtime_size = 13 + 7 * ptr_size
        else:
            # uint32_t sublayer1_workspace_size
            locals_size  = 4
            # 2 Flash ptrs (sublayer1*, sublayer2*) + uint32_t i
            runtime_size = 4 + 2 * ptr_size
        return max(input_workspace_size, base + (locals_size if include_locals else 0) + (runtime_size if include_runtime else 0))


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
            sublayer1_ws = self.sublayer1.get_workspace_size(input_shape, 1)
            sublayer2_ptr = f"(void*)&{var_name}_sublayer2" if self.sublayer2 is not None else "nullptr"
            params_info = [
                (VOID_PTR,  "sublayer1",                f"(void*)&{var_name}_sublayer1"),
                (VOID_PTR,  "sublayer2",                sublayer2_ptr),
                (UINT32_T,  "sublayer1_workspace_size", str(sublayer1_ws)),
            ]
            layer_def += self.get_struct_def(var_name, params_info, QuantizationScheme.NONE, for_arduino)
            layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"
        else:
            if activation_bitwidth == 8:
                quantize_property = ACTIVATION_BITWIDTH_8
            elif activation_bitwidth == 4:
                quantize_property = ACTIVATION_BITWIDTH_4
            elif activation_bitwidth == 2:
                quantize_property = ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(activation_bitwidth)

            branch1_scale      = float(self.branch1_quantize.scale.item())
            branch2_scale      = float(self.branch2_quantize.scale.item())
            output_scale       = float(self.output_quantize.scale.item())
            branch1_zero_point = float(self.branch1_quantize.zero_point.item())
            branch2_zero_point = float(self.branch2_quantize.zero_point.item())
            output_zero_point  = float(self.output_quantize.zero_point.item())
            s1_so = branch1_scale / output_scale
            s2_so = branch2_scale / output_scale
            s1z1  = branch1_scale * branch1_zero_point
            s2z2  = branch2_scale * branch2_zero_point
            z_o   = int(round(output_zero_point - (s1z1 + s2z2) / output_scale))

            data_per_byte = 8 // activation_bitwidth
            sublayer1_ws = self.sublayer1.get_workspace_size(input_shape, data_per_byte)

            sublayer2_ptr = f"(void*)&{var_name}_sublayer2" if self.sublayer2 is not None else "nullptr"
            qp_params_info = [
                (FLOAT_T, "s1_so", f"{s1_so:.9g}f"),
                (FLOAT_T, "s2_so", f"{s2_so:.9g}f"),
                (INT8_T,  "zo",    str(z_o)),
            ]
            layer_def += self.get_packed_struct(f"{var_name}_quantize_params", qp_params_info, for_arduino)
            params_info = [
                (VOID_PTR,  "sublayer1",                f"(void*)&{var_name}_sublayer1"),
                (VOID_PTR,  "sublayer2",                sublayer2_ptr),
                (UINT32_T,  "sublayer1_workspace_size", str(sublayer1_ws)),
                (VOID_PTR,  "quantize_parameters",      f"(void*)&{var_name}_quantize_params"),
                (UINT8_T,   "quantize_property",        quantize_property),
            ]
            layer_def += self.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"


        return layer_header, layer_def, layer_param_def
