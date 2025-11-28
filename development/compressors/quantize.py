"""
@file quantize.py
@brief Quantization Engine (Observer & Application).

This module implements the Quantization Stage of the DMC pipeline.
It manages the transition from Floating Point to Integer representation.

Key Features:
1.  Calibration (Observer): Measures dynamic range (min/max) of activations/weights
    during the `update_parameters` phase.
2.  Fake Quantization (QAT): Simulates quantization noise during retraining to 
    allow the network to adapt (Figure 1: "Retrain Model Parameters").
3.  Real Quantization (Apply): Converts tensors to actual `int8`/`int32` values 
    for the final "Encode Model Parameters" stage.
"""

from typing import Optional, Iterable, Callable, Tuple
from math import prod
from enum import Enum, auto

import torch

from .prune_channel import Prune_Channel
from ..utils import (

    get_quantize_scale_sy,
    get_quantize_scale_zero_point_assy,

    fake_quantize_per_tensor_sy,
    fake_quantize_per_tensor_assy,
    fake_quantize_per_channel_sy, 
    fake_quantize_per_channel_assy,

    quantize_per_tensor_assy,
    quantize_per_tensor_sy,
    quantize_per_channel_assy, 
    quantize_per_channel_sy
)



class QuantizationScheme(Enum):
    NONE = auto()
    DYNAMIC = auto()
    STATIC = auto()

class QuantizationScaleType(Enum):
    SYMMETRIC = auto()
    ASSYMMETRIC = auto()

class QuantizationGranularity(Enum):
    PER_TENSOR = auto()
    PER_CHANNEL = auto()
    

class Quantize:
    """
    Manages quantization state (Scales, Zero-Points) and application logic.
    """
    def __init__(
        self, 
        module,#: Sequential, 
        bitwidth: int, 
        scheme: QuantizationScheme, 
        granularity: QuantizationGranularity, 
        scale_type: QuantizationScaleType, 
        avg_exp: float = 0.01,
        base: Optional[Iterable["Quantize"]] = None, 
        base_accumulator: Optional[Callable[[torch.Tensor, int, Iterable["Quantize"]], torch.Tensor]] = None,
        
        prune_channel: Optional["Prune_Channel"] = None
    ) -> None:
        """
        Initialize Quantizer.

        Args:
            module: The layer being quantized.
            bitwidth: Target precision (e.g., 8, 4, 2 bits).
            scheme: STATIC vs DYNAMIC
            granularity: PER_TENSOR (1 scale per layer) vs PER_CHANNEL (1 scale per filter).
            base: Reference to previous quantizers (used for chaining scales in Static Quantization).
        """
        self.module = module
        self.bitwidth = bitwidth
        self.scheme = scheme
        self.granularity = granularity
        self.scale_type = scale_type
        self.avg_exp = avg_exp
        self.rmin = None
        self.rmax = None

        self.base = base
        
        # Allows a layer to derive its input scale from the previous layer's output scale.
        if base is not None:
            if base_accumulator is None:
                if scale_type == QuantizationScaleType.ASSYMMETRIC:
                    self.base_accumulator: Callable[[Iterable["Quantize"]], Tuple[torch.Tensor, torch.Tensor]] = lambda base : (prod([b.scale for b in base]), sum([b.zero_point for b in base]))
                else:
                    self.base_accumulator: Callable[[Iterable["Quantize"]], torch.Tensor] = lambda base : prod([b.scale for b in base])
            else:
                self.base_accumulator = base_accumulator

        self.prune_channel = prune_channel

    @property
    def scale(self):
        """Calculates the scaling factor: (max - min) / (2^bits - 1)."""
        if self.base is None:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                scale = get_quantize_scale_sy(self.rmax, self.bitwidth)
            else:
                scale = get_quantize_scale_zero_point_assy(self.rmax, self.rmin, self.bitwidth)[0]
        else:
            # Dependent Quantizer (e.g., inherited activation scale)
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                scale = self.base_accumulator(self.base)    
            else:
                scale = self.base_accumulator(self.base)[0]
        return scale
    
    @property
    def zero_point(self):
        """Calculates the integer offset (Zero-Point) for asymmetric quantization."""
        assert self.scale_type == QuantizationScaleType.ASSYMMETRIC, f"scale type should be {QuantizationScaleType.ASSYMMETRIC}"
        if self.base is None:
            return get_quantize_scale_zero_point_assy(self.rmax, self.rmin, self.bitwidth)[1].to(torch.int8)
        else:
            return self.base_accumulator(self.base)[1].to(torch.int8)


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main entry point during Forward Pass.
        
        If training: Updates calibration stats (Min/Max).
        Always: Returns Fake-Quantized tensor (float32 with discrete steps).
        """
        if self.module.training:
            self.update_parameters(x)
        return self.fake_apply(x)
 
    @torch.no_grad()
    def update_parameters(self, x: torch.Tensor) -> None:
        """
        Calibration Phase (Observer).
        
        Tracks the global Min/Max of the incoming tensor 'x' using an 
        Exponential Moving Average (EMA). This ensures the static scale 
        factors are robust to outliers.
        """
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            if self.granularity == QuantizationGranularity.PER_TENSOR:
                if self.rmax is None: self.rmax = x.abs().max()
                else: self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.abs().max() 
            else:
                if self.rmax is None: self.rmax = (x.abs().view(x.size(0), -1).max(dim=1)[0]).to(x.device)
                else: self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.abs().view(x.size(0), -1).max(dim=1)[0]

        else:
            if self.granularity == QuantizationGranularity.PER_TENSOR:
                # Symmetric: Range is [-max(|x|), +max(|x|)]
                if self.rmax is None or self.rmin is None: 
                    self.rmax = x.max()
                    self.rmin = x.min()
                else:
                    self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.max()
                    self.rmin = self.rmin * (1 -self.avg_exp) + self.avg_exp * x.min()
            else:
                # Asymmetric: Range is [min(x), max(x)]
                if self.rmax is None or self.rmin is None: 
                    self.rmax = x.view(x.size(0), -1).max(dim=1)[0]
                    self.rmin = x.view(x.size(0), -1).min(dim=1)[0]
                else:
                    self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.view(x.size(0), -1).max(dim=1)[0]
                    self.rmin = self.rmin * (1 -self.avg_exp) + self.avg_exp * x.view(x.size(0), -1).min(dim=1)[0]
                

    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulates quantization error (Quantization-Aware Training).
        
        Operations:
            x_int = round(x / scale + zero_point)
            x_float = (x_int - zero_point) * scale
        
        This allows backpropagation to adjust weights to be robust to the 
        rounding errors that will occur on the bare-metal device.
        """   
        scale = self.scale
        if self.scale_type == QuantizationScaleType.ASSYMMETRIC:
            zero_point = self.zero_point

        # Handle Pruning Interaction:
        # If per-channel quantization is used on a pruned layer, we must 
        # ensure the scale vector matches the reduced channel count.
        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            assert self.prune_channel is not None, f"The attached layer {self.module} is quantized but no prune_channel was given!"
            
            # Mask out scales for pruned channels to avoid indexing errors
            scale = self.prune_channel.fake_apply(scale)
            scale[scale == 0] = 1
            
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: 
                zero_point = self.prune_channel.fake_apply(zero_point) 
                
        # Apply Fake Quantization
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return fake_quantize_per_tensor_sy(x, scale, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   fake_quantize_per_channel_sy(x, scale, self.bitwidth)
        return fake_quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               fake_quantize_per_channel_assy(x, scale, zero_point, self.bitwidth)


    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs Actual Quantization (Float -> Int).
        
        This generates the Integer Tensors (`int8` or `int32`) that are 
        exported to the C library.
        
        Formula:
            q = clamp(round(x / scale) + zero_point, min_int, max_int)
        """
        # Determine container type based on bitwidth
        dtype = torch.int32 if self.bitwidth > 8 else torch.int8
        
        scale = self.scale
        if self.scale_type == QuantizationScaleType.ASSYMMETRIC:
            zero_point = self.zero_point

        # Handle Pruning (Physical Reduction of Scale Vector)
        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            assert self.prune_channel is not None, f"The attached layer {self.module}is quantized but no prune_channel was given!"
            scale = self.prune_channel.apply(scale)
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: zero_point = self.prune_channel.apply(zero_point) 
        
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return quantize_per_tensor_sy(x, scale, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   quantize_per_channel_sy(x, scale, self.bitwidth, dtype=dtype)
        return quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               quantize_per_channel_assy(x, scale, zero_point, self.bitwidth, dtype=dtype)
    
