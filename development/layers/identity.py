import warnings

from typing import Any, Optional
import torch
from torch import nn

from .layer import Layer


class Identity(Layer, nn.Identity):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)
    

    def init_prune_channel(
        self, 
        sparsity: float, 
        input_shape: torch.Size,
        keep_prev_channel_index: Optional[torch.Tensor], 
        keep_current_channel_index:Optional[torch.Tensor],
        is_output_layer: bool = False, 
        metric: str = "l2"
    ) -> Optional[torch.Tensor]:
        if keep_current_channel_index is not None:
            return keep_current_channel_index
        return keep_prev_channel_index
    
    def get_prune_channel_possible_hyperparameters(self):
        """
        Returns the valid range of channels that can be kept (for Search/NAS).
        Used to generate the Sensitivity Analysis graphs.
        """
        return None


    def init_quantize(
        self, 
        parameter_bitwidth, 
        granularity, scheme, 
        activation_bitwidth=None, 
        previous_output_quantize = None,
        current_output_quantize = None,
    ):

        super().init_quantize(parameter_bitwidth, granularity, scheme, activation_bitwidth, previous_output_quantize)
        if current_output_quantize is None:
            return previous_output_quantize
        warnings.warn(
            (f"{self} recieved a curent_output_quantize, this forces it to use that as the quantization base and not the previous"
            " layer's quantizer, this is likely to using it in a branch with a modifying layer, Linear or Conv.")
        )
        return current_output_quantize


    def get_quantize_possible_hyperparameters(self):
        return super().get_quantize_possible_hyperparameters()

    

    def get_compression_parameters(self):
        pass


    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return 0

    def get_size_in_bits(self) -> int:
        """Calculates the theoretical size of the layer in bits."""
        return 0

    def get_size_in_bytes(self):
        return self.get_size_in_bits() // 8
    
    def get_size_in_KB(self):
        return self.get_size_in_bits() / (8 * 1024)


    def get_output_tensor_shape(self, input_shape):

        return input_shape

    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        raise NotImplementedError("This is not implement because it should have been fused before deployment.")



