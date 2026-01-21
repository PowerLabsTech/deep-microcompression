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


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the pruning mask during the Forward Pass.
        
        Used during the "Fine-tune Parameters". It does NOT 
        reduce tensor size, but multiplies pruned weights by zero so the 
        network learns to function without them.
        """
        if self.layer.training:
            self.update_parameters(x)
        return self.fake_apply(x)
    
    def update_parameters(self, x: torch.Tensor) -> None:
        pass


    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulates pruning by applying a binary mask (Soft Pruning).
        
        Logic:
        1. Create a mask of 1s.
        2. Set pruned channels (Input and Output) to 0.
        3. Multiply weights by mask.
        
        This preserves the tensor shape required by PyTorch's autograd but 
        simulates the sparsity effects.
        """
        if x.ndim > 1:
            assert self.keep_prev_channel_index is not None, "Tensor with ndim > 1 cannot be pruned without keep_prev_channel_index"
            # Removing connections from previous layers
            mask_prev_channel = torch.zeros_like(x)

            mask_prev_channel_index = [slice(None)]*x.ndim
            # PyTorch 2.9 requires multidimensional indices to be tuples; convert to tuple
            # to avoid deprecated non-tuple indexing and future incorrect tensor indexing.
            mask_prev_channel_index[1] = self.keep_prev_channel_index
            mask_prev_channel[tuple(mask_prev_channel_index)] = 1 

            #  Removing connections from this layer
            mask_current_channel = torch.zeros_like(x)

            mask_current_channel_index = [slice(None)]*x.ndim
            mask_current_channel_index[0] = self.keep_current_channel_index

            mask_current_channel[tuple(mask_current_channel_index)] = 1 

            # Combine masks (Intersection of valid inputs and valid outputs)
            mask = torch.mul(mask_current_channel, mask_prev_channel)
        else:
            mask = torch.zeros_like(x)
            mask[self.keep_current_channel_index] = 1

        return torch.mul(x, mask)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs Physical Pruning (Hard Pruning).
        
        Args:
            x: The weight or bias tensor to prune.
            
        Returns:
            A smaller tensor containing only the kept parameters.
        """
        if x.ndim > 1:
            assert self.keep_prev_channel_index is not None, "Tensor with ndim > 1 cannot be pruned without keep_prev_channel_index"
            x = torch.index_select(x, 1, self.keep_prev_channel_index)
            x = torch.index_select(x, 0, self.keep_current_channel_index)
        else:
            x = torch.index_select(x, 0, self.keep_current_channel_index)
        return x