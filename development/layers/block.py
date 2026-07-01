import math
from typing import Optional, Iterable, OrderedDict, Tuple, Union

import torch
from torch import nn
from torch._jit_internal import _copy_to_script_wrapper

from .layer import Layer, Branch
from ..compressors import (
    get_data_bits,
    Quantize,
    QuantizationScheme,
    QuantizationGranularity,
)

from ..utils import (
    pad_bits_to_byte,
    UINT16_T,
    VOID_PTR,
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
        sparsity,  # float | int (uniform) or dict[sublayer_name → float|int] (per-sublayer)
        input_shape: torch.Size,
        keep_prev_channel_index:Optional[torch.Tensor],
        keep_current_channel_index:Optional[torch.Tensor],
        is_output_layer: bool = False,
        metric: str = "l2"
    ):
        def _sparsity(name):
            return sparsity.get(name, 0) if isinstance(sparsity, dict) else sparsity

        layers = list(self.names_layers())
        for name, layer in layers[:-1]:
            keep_prev_channel_index = layer.init_prune_channel(
                _sparsity(name), input_shape, keep_prev_channel_index, keep_current_channel_index,
                is_output_layer=False, metric=metric
            )
            input_shape = layer.get_output_tensor_shape(torch.Size(input_shape))

        name, layer = layers[-1]
        keep_prev_channel_index = layer.init_prune_channel(
            _sparsity(name), input_shape, keep_prev_channel_index, keep_current_channel_index,
            is_output_layer=is_output_layer, metric=metric
        )
        return keep_prev_channel_index


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

        def _parameter_bitwidth(name):
            return parameter_bitwidth.get(name) if isinstance(parameter_bitwidth, dict) else parameter_bitwidth

        def _granularity(name):
            return granularity.get(name) if isinstance(granularity, dict) else granularity
        
        def _activation_bitwidth(name):
            return activation_bitwidth.get(name) if isinstance(activation_bitwidth, dict) else activation_bitwidth

        if scheme != QuantizationScheme.STATIC:
            for name, layer in self.names_layers():
                layer.init_quantize(_parameter_bitwidth(name), _granularity(name), scheme)
            return

        # Only the last layer receives current_output_quantize. That argument forces a layer's
        # output scale to match an external quantizer — required when this Block sits inside a
        # Branch, where both paths must share the same integer scale for the residual add to be
        # valid. Intermediate layers must calibrate independently; passing the constraint inward
        # would force every layer's output to the external scale, which is wrong (activations
        # mid-block should not be clamped to the final branch scale) and would prevent each
        # layer from finding its own optimal range during QAT.

        layers = list(self.names_layers())
        for name, layer in layers[:-1]:
            previous_output_quantize = layer.init_quantize(
                _parameter_bitwidth(name), _granularity(name),
                scheme, _activation_bitwidth(name), previous_output_quantize,
                current_output_quantize=None
            )

        name, layer = layers[-1]
        previous_output_quantize = layer.init_quantize(
            _parameter_bitwidth(name), _granularity(name),
            scheme, _activation_bitwidth(name), previous_output_quantize,
            current_output_quantize=current_output_quantize
        )

        if hasattr(self[-1], "output_quantize"):
            assert self[-1].output_quantize is previous_output_quantize

        return previous_output_quantize

    
    def get_prune_channel_possible_hyperparameters(self):
        result = {
            name: hp
            for name, layer in self.names_layers()
            if (hp := layer.get_prune_channel_possible_hyperparameters()) is not None
        }
        return result if result else None

    def get_quantize_possible_hyperparameters(self):
        result = {
            name: hp
            for name, layer in self.names_layers()
            if (hp := layer.get_quantize_possible_hyperparameters()) is not None
        }
        return result if result else None

    def is_quantize_config_valid(
        self,
        parameter_bitwidth,
        granularity,
        activation_bitwidth,
        raise_error: bool = True,
    ) -> bool:
        """
        Validates and fills per-sublayer quantization config dicts in-place.

        Called by Sequential.is_compression_config_valid for Block entries whose
        parameter_bitwidth or activation_bitwidth is a per-sublayer dict.  Scalar
        (uniform) arguments are accepted as-is without expansion.
        """
        valid_names = set(self.names())

        # Validate activation_bitwidth first, since it is independent of parameter_bitwidth and granularity
        if isinstance(activation_bitwidth, dict):
            for sub_name in activation_bitwidth:
                if sub_name not in valid_names:
                    if raise_error:
                        raise NameError(
                            f"Unknown sublayer '{sub_name}' in activation_bitwidth for Block; "
                            f"valid names: {sorted(valid_names)}"
                        )
                    return False
            for sub_name, ab in activation_bitwidth.items():
                if isinstance(ab, dict):
                    if not isinstance(self[sub_name], (Block, Branch)):
                        if raise_error:
                            raise TypeError(
                                f"Block activation_bitwidth['{sub_name}'] is a dict but '{sub_name}' is not a container layer"
                            )
                        return False
                elif not isinstance(ab, int) or ab not in [2, 4, 8]:
                    if raise_error:
                        raise ValueError(
                            f"Block activation_bitwidth['{sub_name}'] must be int in [2, 4, 8], got {ab!r}"
                        )
                    return False
            for sub_name, layer in self.names_layers():
                if sub_name not in activation_bitwidth:
                    hp = layer.get_quantize_possible_hyperparameters()
                    if hp is not None and "activation_bitwidth" in hp:
                        activation_bitwidth[sub_name] = 8

        # parameter_bitwidth and granularity are always paired — validate them together
        if (isinstance(parameter_bitwidth, int) and not isinstance(granularity, QuantizationGranularity)) or \
           (isinstance(granularity, QuantizationGranularity) and not isinstance(parameter_bitwidth, int)):
            if raise_error:
                raise ValueError("Block: parameter_bitwidth and granularity must both be scalars or both be dicts.")
            return False

        if (isinstance(parameter_bitwidth, dict) and not isinstance(granularity, dict)) or \
           (isinstance(granularity, dict) and not isinstance(parameter_bitwidth, dict)):
            if raise_error:
                raise ValueError("Block: parameter_bitwidth and granularity must both be dicts or both be scalars.")
            return False

        if isinstance(parameter_bitwidth, dict):
            if parameter_bitwidth.keys() != granularity.keys():
                if raise_error:
                    raise ValueError("Block: parameter_bitwidth and granularity dicts must have the same keys.")
                return False

            for (sub_name, pb), (_, gran) in zip(parameter_bitwidth.items(), granularity.items()):
                if sub_name not in valid_names:
                    if raise_error:
                        raise NameError(
                            f"Unknown sublayer '{sub_name}' in parameter_bitwidth for Block; "
                            f"valid names: {sorted(valid_names)}"
                        )
                    return False
                if isinstance(pb, dict):
                    if not isinstance(self[sub_name], (Block, Branch)):
                        if raise_error:
                            raise TypeError(
                                f"Block parameter_bitwidth['{sub_name}'] is a dict but '{sub_name}' is not a container layer"
                            )
                        return False
                else:
                    if not isinstance(pb, int) or pb not in [2, 4, 8]:
                        if raise_error:
                            raise ValueError(
                                f"Block parameter_bitwidth['{sub_name}'] must be int in [2, 4, 8], got {pb!r}"
                            )
                        return False
                    if not isinstance(gran, QuantizationGranularity):
                        if raise_error:
                            raise TypeError(
                                f"Block granularity['{sub_name}'] must be QuantizationGranularity, got {type(gran)}"
                            )
                        return False

            for sub_name in valid_names:
                if sub_name not in parameter_bitwidth:
                    parameter_bitwidth[sub_name] = 8
                    granularity[sub_name] = QuantizationGranularity.PER_CHANNEL

        # Single validation pass for Block/Branch sublayer configs.
        # Both dicts are fully filled here, so ab and pb are available together.
        for sub_name, sub_layer in self.names_layers():
            if isinstance(sub_layer, (Block, Branch)):
                ab_entry   = activation_bitwidth.get(sub_name) if isinstance(activation_bitwidth, dict) else activation_bitwidth
                pb_entry   = parameter_bitwidth.get(sub_name) if isinstance(parameter_bitwidth, dict) else parameter_bitwidth
                gran_entry = granularity.get(sub_name) if isinstance(granularity, dict) else granularity
                if isinstance(ab_entry, dict) or isinstance(pb_entry, dict):
                    if not sub_layer.is_quantize_config_valid(pb_entry, gran_entry, ab_entry, raise_error):
                        return False

        return True

    def is_prune_config_valid(self, sparsity, raise_error: bool = True) -> bool:
        """
        Validates per-sublayer sparsity dict and fills defaults (0 for missing entries).
        Scalar sparsity is not expanded here — Sequential handles uniform values.
        """
        if not isinstance(sparsity, dict):
            return True
        valid_names = set(self.names())
        for sub_name in sparsity:
            if sub_name not in valid_names:
                if raise_error:
                    raise NameError(
                        f"Unknown sublayer '{sub_name}' in sparsity for Block; "
                        f"valid names: {sorted(valid_names)}"
                    )
                return False
        for sub_name, s in sparsity.items():
            if not isinstance(s, (float, int)):
                if raise_error:
                    raise TypeError(f"Block sparsity['{sub_name}'] must be float or int, got {type(s)}")
                return False
        for sub_name in valid_names:
            if sub_name not in sparsity:
                sparsity[sub_name] = 0
        return True

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
    

    def get_workspace_size(
        self, input_shape, include_locals=False,
        include_runtime=False, ptr_size=2
    ) -> int:
        if isinstance(input_shape, tuple): input_shape = torch.Size(input_shape)
        output_shape = input_shape
        data_bits = get_data_bits(self)
        max_sublayer_size = pad_bits_to_byte(input_shape.numel() * data_bits)

        for layer in self.layers():
            size = layer.get_workspace_size(
                torch.Size(output_shape), include_locals, include_runtime, ptr_size
            )
            max_sublayer_size = max(max_sublayer_size, size)
            output_shape = layer.get_output_tensor_shape(output_shape)
        if not (include_locals or include_runtime):
            return max_sublayer_size
        # uint16_t num_layers read from buffer
        block_locals  = 2 if include_locals  else 0
        # uint16_t i loop counter + Layer* local ptr
        block_runtime = (2 + ptr_size) if include_runtime else 0
        return max_sublayer_size + block_locals + block_runtime


    @torch.no_grad()
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        layer_header, layer_def, layer_param_def = "", "", ""

        for layer_name, layer in self.names_layers():
            layer_var_name = f"{var_name}_{layer_name}"
            layer_header_, layer_def_, layer_param_def_ = layer.convert_to_c(layer_var_name, input_shape, for_arduino=for_arduino)
            input_shape = layer.get_output_tensor_shape(torch.Size(input_shape))

            layer_header += layer_header_
            layer_def += layer_def_
            layer_param_def += layer_param_def_

        scheme = None
        if self.is_quantized:
            scheme = self.__dict__["_dmc"]["quantize"]["scheme"]

        params_info = [(UINT16_T, "num_layers", str(len(self)))]
        for layer_name, _ in self.names_layers():
            params_info.append((VOID_PTR, layer_name, f"(void*)&{var_name}_{layer_name}"))

        if scheme == QuantizationScheme.STATIC:
            layer_def += self.get_struct_def(var_name, params_info, QuantizationScheme.STATIC, for_arduino)
            layer_header += f"extern {self.__class__.__name__}_SQ {var_name};\n\n"
        else:
            layer_def += self.get_struct_def(var_name, params_info, QuantizationScheme.NONE, for_arduino)
            layer_header += f"extern {self.__class__.__name__} {var_name};\n\n"

        return layer_header, layer_def, layer_param_def
    

