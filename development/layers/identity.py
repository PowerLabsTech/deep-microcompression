from typing import Any, Optional

import torch
from torch import nn

from .layer import Layer


class Identity(Layer, nn.Identity):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)

    # init_prune_channel, get_prune_channel_possible_hyperparameters,
    # init_quantize, get_quantize_possible_hyperparameters all use the
    # base class passthrough defaults — no overrides needed.

    def get_compression_parameters(self):
        pass

    def get_workspace_size(self, input_shape, data_per_byte) -> int:
        return 0

    def get_size_in_bits(self) -> int:
        return 0

    def get_output_tensor_shape(self, input_shape):
        return input_shape

    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        raise NotImplementedError(
            "Identity.convert_to_c is not implemented: Identity layers should have "
            "been fused before deployment. Call model.fuse() first."
        )
