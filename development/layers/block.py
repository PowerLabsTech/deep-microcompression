from typing import Optional, Iterable, OrderedDict, Tuple, Union

import torch
from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

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

class Block(Layer, nn.Module):

    def __init__(self, *args):
        super().__init__()

        self.class_idx = dict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            # Auto-name layers with type_index convention (e.g. conv2d_0)
            for layer in args:
                if isinstance(layer, Layer) or isinstance(layer, nn.Module): 
                    idx = self.class_idx.get(layer.__class__.__name__, -1) + 1
                    self.class_idx[layer.__class__.__name__] = idx
                    layer_type = layer.__class__.__name__.lower()
                    self.add_module(f"{layer_type}_{idx}", layer) # type: ignore
                else:
                    raise TypeError(f"layer of type {type(layer)} isn't a Layer or Module.")



    def names_layers(self)-> Iterable[Tuple[str, Layer]]:
        """
        Yields (name, layer) pairs.
        """
        for name, layer in self._modules.items():
            yield name, layer

    def names(self) -> Iterable[str]:
        for name in self._modules.keys():
            yield name

    def layers(self) -> Iterable[Layer]:
        for layer in self._modules.values():
            yield layer
    

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, str, int]) -> Union["Block", Layer]:
        """
        Access layers by index, name, or slice.
        """
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        elif isinstance(idx, str):
            return self._modules[idx]
        elif isinstance(idx, int):
            lenght = len(self)
            if -lenght <= idx < lenght:
                idx %= lenght
                return self[list(self.names())[idx]]
            raise IndexError(f"index {idx} is out of range")
        else:
            raise IndexError(f"Unknown index {idx}")


    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)


    def forward(self, input):
        for layer in self.layers():
            input = layer(input)
        return input
    

    def init_prune_channel(
        self, 
        sparsity: float, 
        input_shape: torch.Size,
        keep_prev_channel_index:Optional[torch.Tensor], 
        keep_current_channel_index:Optional[torch.Tensor],
        is_output_layer: bool = False, 
        metric: str = "l2"
    ):
        
        # Prune all layers except last
        for name, layer in list(self.names_layers())[:-1]:
            keep_prev_channel_index = layer.init_prune_channel(
                sparsity, input_shape, keep_prev_channel_index, keep_current_channel_index,
                # sparsity[name], input_shape, keep_prev_channel_index, keep_current_channel_index,
                is_output_layer=False, metric=metric
            )
            input_shape = layer.get_output_tensor_shape(torch.Size(input_shape))

        name, layer = list(self.names_layers())[-1]
        keep_prev_channel_index = layer.init_prune_channel(
            sparsity, input_shape, keep_prev_channel_index, keep_current_channel_index,
            # sparsity[name], input_shape, keep_prev_channel_index, keep_current_channel_index,
            is_output_layer=is_output_layer, metric=metric
        )

        return keep_prev_channel_index



    def get_prune_channel_possible_hyperparameters(self):
        return None
    
    def get_quantize_possible_hyperparameters(self):
        return None



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
            for layer in self.layers():
                layer.init_quantize(parameter_bitwidth, scheme, granularity)
            return

        previous_output_quantize = self.input_quantize
        for layer in self.layers():
            previous_output_quantize = layer.init_quantize(bitwidth, scheme, granularity, previous_output_quantize)

        if hasattr(self[-1], "output_quantize"):
            return self[-1].output_quantize 
        return None


    
    def get_prune_channel_possible_hyperparameters(self):
        return self.sublayer1.get_prune_channel_possible_hyperparameters()


    def get_compression_parameters(self):
        return


    def get_size_in_bits(self) -> int:

        size = 0
        for layer in self.layers():
            size += layer.get_size_in_bits()
        return size


    def get_output_tensor_shape(self, input_shape):
        for layer in self.layers():
            input_shape = layer.get_output_tensor_shape(input_shape)
        return input_shape
    

    def get_workspace_size(self, input_shape, data_per_byte):
        output_shape= input_shape
        max_layer_acitivation_workspace_size = 0

        for layer in self.layers():
            workspace_size = layer.get_workspace_size(torch.Size(output_shape), data_per_byte)
            max_layer_acitivation_workspace_size = max(max_layer_acitivation_workspace_size, workspace_size)
            output_shape = layer.get_output_tensor_shape(output_shape)
        return max_layer_acitivation_workspace_size




    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        layer_header, layer_def, layer_param_def = "", "", ""

        layers_header = (
            f"extern Layer* {var_name}_layers[{len(self)}];\n"
            f"extern {self.__class__.__name__} {var_name};\n\n"
        )
    
        layers_def = (
            f"{self.__class__.__name__} {var_name}({var_name}_layers, {len(self)});\n"
            f"Layer* {var_name}_layers[{len(self)}] = {{\n"
        )
        
        for layer_name, layer in self.names_layers():
            layer_var_name = f"{var_name}_{layer_name}"

            layers_def += f"    &{layer_var_name},\n"
            layer_header_, layer_def_, layer_param_def_ = layer.convert_to_c(f"{layer_var_name}", input_shape, for_arduino=for_arduino)
            input_shape = layer.get_output_tensor_shape(torch.Size(input_shape))  

            layer_header += layer_header_
            layer_def += layer_def_
            layer_param_def += layer_param_def_

        layers_def += "};\n\n"

        layer_def += layers_def

        layer_header += layers_header
        
        return layer_header, layer_def, layer_param_def
    

