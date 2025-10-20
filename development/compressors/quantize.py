from typing import Optional, Iterable, Callable, Tuple
from math import prod

import torch

from .prune_channel import Prune_Channel
from ..utils import (
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,

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



class Quantize:

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

        self.module = module
        self.bitwidth = bitwidth
        self.scheme = scheme
        self.granularity = granularity
        self.scale_type = scale_type
        self.avg_exp = avg_exp
        self.rmin = None
        self.rmax = None

        self.base = base

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
        if self.base is None:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                scale = get_quantize_scale_sy(self.rmax, self.bitwidth)
            else:
                scale = get_quantize_scale_zero_point_assy(self.rmax, self.rmin, self.bitwidth)[0]
        else:
            if self.scale_type == QuantizationScaleType.SYMMETRIC:
                scale = self.base_accumulator(self.base)    
            else:
                scale = self.base_accumulator(self.base)[0]
        return scale
    
    @property
    def zero_point(self):
        assert self.scale_type == QuantizationScaleType.ASSYMMETRIC, f"scale type should be {QuantizationScaleType.ASSYMMETRIC}"
        if self.base is None:
            return get_quantize_scale_zero_point_assy(self.rmax, self.rmin, self.bitwidth)[1].to(torch.int8)
        else:
            return self.base_accumulator(self.base)[1].to(torch.int8)


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.module.training:
            self.update_parameters(x)
        return self.fake_apply(x)
 
    @torch.no_grad()
    def update_parameters(self, x: torch.Tensor) -> None:
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            if self.granularity == QuantizationGranularity.PER_TENSOR:
                if self.rmax is None: self.rmax = x.abs().max()
                else: self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.abs().max() 
            else:
                if self.rmax is None: self.rmax = (x.abs().view(x.size(0), -1).max(dim=1)[0]).to(x.device)
                else: self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.abs().view(x.size(0), -1).max(dim=1)[0]

            # print("the percent of x.max() is ", (x.abs() <= x.max()).sum().item()/x.numel(), self.module)
            # print("effect of abs", x.abs().max() == x.max().abs() or x.abs().max() == x.min().abs(), x.min().abs() == x.abs().max())
        else:
            if self.granularity == QuantizationGranularity.PER_TENSOR:
                if self.rmax is None or self.rmin is None: 
                    self.rmax = x.max()
                    self.rmin = x.min()
                else:
                    self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.max()
                    self.rmin = self.rmin * (1 -self.avg_exp) + self.avg_exp * x.min()
            else:
                if self.rmax is None or self.rmin is None: 
                    self.rmax = x.view(x.size(0), -1).max(dim=1)[0]
                    self.rmin = x.view(x.size(0), -1).min(dim=1)[0]
                else:
                    self.rmax = self.rmax * (1 -self.avg_exp) + self.avg_exp * x.view(x.size(0), -1).max(dim=1)[0]
                    self.rmin = self.rmin * (1 -self.avg_exp) + self.avg_exp * x.view(x.size(0), -1).min(dim=1)[0]
                

    def fake_apply(self, x: torch.Tensor) -> torch.Tensor:
               
        scale = self.scale
        if self.scale_type == QuantizationScaleType.ASSYMMETRIC:
            zero_point = self.zero_point

        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            assert self.prune_channel is not None, f"The attached layer {self.module}is quantized but no prune_channel was given!"
            scale = self.prune_channel.fake_apply(scale)
            scale[scale == 0] = 1
            
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: 
                zero_point = self.prune_channel.fake_apply(zero_point) 
        
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return fake_quantize_per_tensor_sy(x, scale, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   fake_quantize_per_channel_sy(x, scale, self.bitwidth)
        return fake_quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               fake_quantize_per_channel_assy(x, scale, zero_point, self.bitwidth)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.int32 if self.bitwidth > 8 else torch.int8
        
        scale = self.scale
        if self.scale_type == QuantizationScaleType.ASSYMMETRIC:
            zero_point = self.zero_point

        if self.granularity == QuantizationGranularity.PER_CHANNEL and self.module.is_pruned_channel:
            assert self.prune_channel is not None, f"The attached layer {self.module}is quantized but no prune_channel was given!"
            scale = self.prune_channel.apply(scale)
            if self.scale_type == QuantizationScaleType.ASSYMMETRIC: zero_point = self.prune_channel.apply(zero_point) 
        
        if self.scale_type == QuantizationScaleType.SYMMETRIC:
            return quantize_per_tensor_sy(x, scale, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
                   quantize_per_channel_sy(x, scale, self.bitwidth, dtype=dtype)
        return quantize_per_tensor_assy(x, scale, zero_point, self.bitwidth, dtype=dtype) if self.granularity == QuantizationGranularity.PER_TENSOR else \
               quantize_per_channel_assy(x, scale, zero_point, self.bitwidth, dtype=dtype)
    
