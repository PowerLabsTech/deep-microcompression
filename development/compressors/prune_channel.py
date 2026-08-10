"""
@file prune_channel.py
@brief Structural Pruning Implementation (Masking & Reduction).

This module implements the Structured Pruning Stage of the 
DMC pipeline. It handles the logic for removing entire groups of parameters 
(channels/filters) rather than individual weights.

The class provides two modes of operation:
1.  Fake Apply (Masking): Zeroes out weights during the "Fine-tune Parameters" 
2.  Apply (Actual Reduction): Slices the tensors to produce the Significant Model Reduction 
    required for the final C-code generation.
"""

from typing import Optional
from math import prod

import torch

class Prune_Channel:
    """
    Manages channel pruning masks and physical tensor reduction.
    
    It ensures that when a filter is pruned in Layer N, the corresponding input 
    channel in Layer N+1 is also handled (Dependency Propagation).
    """
    def __init__(
        self, 
        layer, 
        keep_current_channel_index: torch.Tensor, 
        keep_prev_channel_index: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Pruning Controller.

        Args:
            layer: The layer being pruned.
            keep_current_channel_index: Indices of Output Channels (Filters) to keep.
                                        Determines the layer's output dimension.
            keep_prev_channel_index: Indices of Input Channels to keep.
                                     Must align with the previous layer's remaining filters.
        """
        self.layer = layer
        self.keep_current_channel_index = keep_current_channel_index
        self.keep_prev_channel_index = keep_prev_channel_index
        # Mask is constant for the lifetime of this pruner; cache it on first use.
        self._mask_cache: Optional[torch.Tensor] = None


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the pruning mask during the Forward Pass.

        Does NOT reduce tensor size — multiplies pruned weights by zero so the
        network learns to function without them (Soft Pruning / Fine-tune stage).
        """
        if self.layer.training:
            self.update_parameters(x)
        return self.fake_apply(x)


    def update_parameters(self, x: torch.Tensor) -> None:
        """
        Not used in this implementation, but can be extended to update internal state or perform additional checks during training.
        """
        pass


    def _build_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Builds and caches the binary pruning mask matching x's shape."""
        device = x.device
        if x.ndim > 1:
            assert self.keep_prev_channel_index is not None, \
                "Tensor with ndim > 1 cannot be pruned without keep_prev_channel_index"
            mask = torch.zeros_like(x)
            out_idx = self.keep_current_channel_index.to(device)
            in_idx  = self.keep_prev_channel_index.to(device)
            mask[out_idx[:, None], in_idx[None, :]] = 1
        else:
            mask = torch.zeros_like(x)
            mask[self.keep_current_channel_index.to(device)] = 1
        return mask


    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulates pruning by zeroing out pruned channels (Soft Pruning).

        The mask is built once and reused — indices are fixed after init_prune_channel.
        """
        if self._mask_cache is None or self._mask_cache.shape != x.shape:
            self._mask_cache = self._build_mask(x)
        return torch.mul(x, self._mask_cache)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs Physical Pruning (Hard Pruning).

        Args:
            x: The weight or bias tensor to prune.

        Returns:
            A smaller tensor containing only the kept parameters.
        """
        device = x.device
        if x.ndim > 1:
            assert self.keep_prev_channel_index is not None, "Tensor with ndim > 1 cannot be pruned without keep_prev_channel_index"
            x = torch.index_select(x, 1, self.keep_prev_channel_index.to(device))
            x = torch.index_select(x, 0, self.keep_current_channel_index.to(device))
        else:
            x = torch.index_select(x, 0, self.keep_current_channel_index.to(device))
        return x