"""
@file sequential.py
@brief DMC Pipeline Container: Orchestrates Pruning, Quantization, and Code Gen.

This module implements the core container for the Deep Microcompression (DMC) 
framework. It aligns with the "Model Development" phase described in Section III 
and Figure 1 of the paper.

Pipeline Stages Managed by this Container:
1.  Structured Pruning: Storage of channel masks and management of dense 
    output generation.
2.  Quantization: Handling of Static Quantization parameters (scale/zero-point) 
    to enable integer-only arithmetic.
3.  Deployment: Acts as the source of truth for generating the dependency-free 
    C library.
"""

__all__ = [
    "Sequential"
]
import copy, math, itertools
import os
import warnings
from typing import (
    List, Tuple, Dict, OrderedDict, Iterable, Callable, Optional, Union, Any
)
from tqdm.auto import tqdm

import torch
from torch import nn
from torch._jit_internal import _copy_to_script_wrapper
from torch.utils import data

# DMC Framework Imports
from ..layers.layer import Layer
from ..layers.activation import ReLU, ReLU6
from ..layers.batchnorm import BatchNorm2d
from ..layers.block import Block
from ..layers.branch import Branch
from ..layers.conv import Conv2d
from ..layers.flatten import Flatten
from ..layers.linear import Linear
from ..layers.padding import ConstantPad2d
from ..layers.pooling import AvgPool2d, MaxPool2d
from ..utils import get_data_bits


from ..compressors import (
    Quantize,
    QuantizationScheme,
    QuantizationScaleType,
    QuantizationGranularity,
    QuantizationBitWidthError,
)
from ..utils import (
    convert_tensor_to_bytes_var,
    pad_bits_to_byte,

    ACTIVATION_BITWIDTH_2,
    ACTIVATION_BITWIDTH_4,
    ACTIVATION_BITWIDTH_8,
    INT8_T,
    UINT8_T,
    UINT32_T,
    VOID_PTR,
    FLOAT_T
)
from .fuse import *

class Sequential(nn.Sequential):
    """Extended Sequential container with additional functionality for:
        - Quantization (dynamic/static, per-tensor/per-channel)
        - Pruning
        - Training utilities
        - C code generation
    """

    _modules: dict[str, Layer]

    def __init__(self, *args):
        """Initialize Sequential model with automatic layer naming
        
        Args:
            *args: Can be either:
                - An OrderedDict of layers
                - Individual layer instances
        """
        super(Sequential, self).__init__()
        setattr(self, "_dmc", dict())

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

        self.fit_history = dict()

        self.is_pruned_channel = False
        self.is_quantized = False

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
    def __getitem__(self, idx: Union[slice, str, int]) -> Union["Sequential", Layer]:
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
        
    def __add__(self, other) -> "Sequential":

        result = Sequential()
        for name, layer in self.names_layers():
            result.add_layer(name=name, layer=layer)        
        
        if isinstance(other, Sequential):
            for layer in other:
                result.add_layer(layer=layer)        
            return result
        elif isinstance(other, (Layer, nn.Module)):
            result.add_layer(other)
            return result
        raise RuntimeError(f"cannot add type {other} to Sequential")
    
    def add_layer(self, layer: Union[Layer, nn.Module], name: str="") -> None:
        """
        Adds a layer and preserves the naming layer_idx, 
        internally it uses the _modules container for to store the layers.
        """
        if (not name):
            idx = self.class_idx.get(layer.__class__.__name__, -1) + 1
            layer_type = layer.__class__.__name__.lower()
            name = f"{layer_type}_{idx}"
        else:
            layer_type, idx = name.split("_")
            try:
                idx = int(idx)
                old_idx = self.class_idx.get(layer.__class__.__name__, -1)
                assert idx > old_idx, f"Adding layer {name} to self will over-write the another layer."
            except ValueError:
                # Do not update the class idx if idx is not an int
                pass
            
        self.class_idx[layer.__class__.__name__] = idx
        return super().add_module(name, layer) 
        

    @property
    def is_compressed(self) -> bool:
        """
        Indicates if the model has entered the compressed pipeline.
        """
        return self.is_pruned_channel or self.is_quantized

    @property
    def output_quantize(self) -> Optional[Quantize]:
        """
        Retrieves the Quantization Parameters (Scale, Zero-Point) for the 
        final output tensor.
        
        Mapping to Paper Section III-B (Quantization):
        In 'Static Quantization', the final integer output of the network must 
        be de-quantized or interpreted by the application. This property exposes 
        the necessary parameters to the C-generation engine so the microcontroller 
        knows how to interpret the final inference result.
        
        Returns:
            Quantization parameters if scheme is STATIC, else None.
        """
        if self.is_quantized and self.__dict__["_dmc"]["compression_config"]["quantize"]["scheme"] == QuantizationScheme.STATIC:
            return self[-1].output_quantize
        return None
    
    @output_quantize.setter
    def output_quantize(self, _: Any):
        # Read-only property derived from the final layer's state
        pass


    def forward(self, input:torch.Tensor):
        """
        Forward pass executing the DMC inference pipeline.
        
        This method adapts its behavior based on the pipeline stage:
        1.  Full Precision: Standard floating-point inference.
        2.  Quantization-Aware (Stage 2): Simulates integer arithmetic constraints
            (Figure 2) by applying fake-quantization to inputs and activations.
                
        Args:
            input: Input tensor (Float32).
            
        Returns:
            Output tensor (simulating integer output if quantized).
        """
        # DMC Stage 2: Quantization-Aware Inference Simulation
        # If in quantized mode, explicitly quantize the input data first.
        # This simulates the "Input Data -> Quantize" step implicit in integer-only hardware
        # where sensors/inputs must be converted to int8 before processing.
        if self.is_quantized:
            if hasattr(self, "input_quantize"):
                input = self.input_quantize(input)

        for i, layer in enumerate(self):
            # Execute layers sequentially.
            # If is_quantized=True, individual layers (Conv2d, Linear) execute 
            # their own fake-quantization logic (Weight/Activation quantization)
            input = layer(input)
        return input


    def fit(
        self, 
        train_dataloader: data.DataLoader, 
        epochs: int, 
        criterion_fun: torch.nn.Module, 
        optimizer_fun: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        validation_dataloader: Optional[data.DataLoader] = None, 
        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {},
        verbose: bool = True,
        progress: bool = True,
        callbacks: List[Callable] = [],
        batch_size = 32,
        device: str = "cpu"
    ) -> Dict[str, List[float]]:
        """
        Universal Training Loop for DMC Optimization Stages.
        
        This method implements the retraining logic required for two critical 
        steps in the DMC Development Pipeline (Figure 1):
        
        1.  "Fine-tune Parameters" (Post-Pruning): After Structured Channel Pruning (Section III-A), the model is retrained 
            to recover accuracy lost due to removing filters.
            
        2.  "Retrain Model Parameters" (QAT): During Quantization (Section III-B), this loop performs Quantization-Aware 
            Training (QAT) to simulate quantization noise and adapt weights for 
            low-bitwidth (e.g., 2-bit/4-bit) representation.
        
        Args:
            train_dataloader: Source of training data.
            epochs: Duration of retraining/fine-tuning.
            criterion_fun: Loss function (e.g., CrossEntropy).
            optimizer_fun: Optimizer (e.g., SGD/Adam).
            validation_dataloader: Validation set for monitoring accuracy recovery.
            metrics: Dict of metric functions (e.g., {'acc': accuracy_fn}).
            device: Execution target ('cpu' or 'cuda').
            
        Returns:
            Dictionary containing loss and metric history for analysis.
        """
        history = dict()
        metrics_values = dict()

        for epoch in tqdm(range(epochs), desc=f"DMC Training (Epochs 1-{epochs})", disable=not progress, leave=False):
            # Training phase
            train_loss = 0
            train_data_len = 0
            metrics_result = {name : 0. for name in metrics.keys()}

            self.train()
            for X, y_true in train_dataloader:
                X = X.to(device)
                y_true = y_true.to(device)

                optimizer_fun.zero_grad()
                y_pred = self(X)
                loss = criterion_fun(y_pred, y_true)
                loss.backward()
                optimizer_fun.step()

                train_loss += loss.item() * X.size(0)
                train_data_len += X.size(0)

                with torch.inference_mode():
                    for name, metric_func in metrics.items():
                        metrics_result[name] += metric_func(y_pred.detach(), y_true) * X.size(0)

            train_loss /= train_data_len
            for name in metrics.keys():
                metrics_values[f"train_{name}"] = metrics_result[name] / train_data_len


            # Validation phase
            if validation_dataloader is not None:
                self.eval()
                metrics_result = self.evaluate(validation_dataloader, metrics | {"loss": criterion_fun}, device=device)
                validation_loss = metrics_result["loss"]
                for name in metrics.keys():
                    metrics_values[f"validation_{name}"] = metrics_result[name]


                # ReduceLROnPlateau needs the validation metric
                if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(validation_loss)
            
            # Epoch-based schedulers (CosineAnnealingLR, StepLR, etc.) step every epoch
            if lr_scheduler is not None and not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

            # Logging
                if verbose:
                    print(f"epoch {epoch:4d} | train loss {train_loss:.4f} | validation loss {validation_loss:.4f}", end="")
                    if metrics is not None: 
                        for name in metrics:
                            print(f" | train {name} {metrics_values[f'train_{name}']:.4f} | validation {name} {metrics_values[f'validation_{name}']:.4f}", end="")
                    print()

                # Store validation metrics
                self.fit_history["validation_loss"] = self.fit_history.get("validation_loss", []) + [validation_loss]
                history["validation_loss"] = history.get("validation_loss", []) + [validation_loss]
                if metrics is not None: 
                    for name in metrics:
                        self.fit_history[f"validation_{name}"] = self.fit_history.get(f"validation_{name}", []) + [metrics_values[f"validation_{name}"]]
                        history[f"validation_{name}"] = history.get(f"validation_{name}", []) + [metrics_values[f"validation_{name}"]]

            # if validation_dataloader is None:
            else:
                if verbose:
                    print(f"epoch {epoch:4d} | train loss {train_loss:.4f}", end="")
                    if metrics is not None:
                        for name in metrics:
                            print(f" | train {name} {metrics_values[f'train_{name}']:.4f}", end="")
                    print()
            
            # Store training metrics
            self.fit_history["train_loss"] = self.fit_history.get("train_loss", []) + [train_loss]
            history["train_loss"] = history.get("train_loss", []) + [train_loss]
            if metrics is not None: 
                for name in metrics:
                    self.fit_history[f"train_{name}"] = self.fit_history.get(f"train_{name}", []) + [metrics_values[f"train_{name}"]]
                    history[f"train_{name}"] = history.get(f"train_{name}", []) + [metrics_values[f"train_{name}"]]
            # Callbacks (e.g., EarlyStopping, ModelCheckpoint)
            for callback in callbacks:
                if callback(self, history, epoch):
                    return history

        return history


    @torch.inference_mode()
    def evaluate(
        self, 
        data_loader: data.DataLoader, 
        metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]],
        progress: bool = True,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
 
        Args:
            data_loader: Validation/Test dataset loader.
            metrics: Dictionary of metric functions (e.g. Top-1 Accuracy).
            device: Execution target.
            
        Returns:
            Dictionary of calculated metrics.
        """
        metric_results = dict()
        data_len = 0
        for metric_name in metrics.keys():
            metric_results[metric_name] = 0
            
        self.eval()
        for X, y_true in tqdm(data_loader, desc="Evaluating", leave=False, disable=not progress):
            X = X.to(device)
            y_true = y_true.to(device)
            y_pred = self(X)
            for metric_name, metric_func in metrics.items():
                metric_results[metric_name] += metric_func(y_pred, y_true) * X.size(0)
            data_len += X.size(0)

        for metric_name in metrics.keys():
            metric_results[metric_name] /= data_len
        return metric_results
    

    def init_compress(
        self,
        config: Dict,
        input_shape: Tuple,
        calibration_data: Optional[torch.Tensor] = None,
        force_prune_channel:bool=False,
        device:Union[str, torch.device] = torch.device("cpu")
    ) -> "Sequential":
        """
        Initializes the Deep Microcompression (DMC) Pipeline.
        
        This entry point transforms the standard model into a DMC-optimized
        candidate by activating the specific pipeline stages described in Section III.
        
        It handles the transition between:
        - Baseline Model 
        - Pruned Model 
        - Quantized Model 
        
        Args:
            config: Dictionary defining specific parameters for Pruning (e.g., sparsity)
                    and Quantization (e.g., 4-bit Static).
            input_shape: Input tensor dimensions. Critical for:
                         1. Propagating pruning masks (calculating input/output channels).
                         2. Generating C-code buffer sizes.
            calibration_data: A representative dataset (DataLoader).
                              MANDATORY for Static Quantization (Section III-B) to 
                              pre-calculate min/max ranges for integer scaling factors.
                              
        Returns:
            A new, independent Sequential instance configured for the requested 
            compression pipeline.
        """
        config = copy.deepcopy(config)
        # Validate configuration against supported DMC schemes
        if not self.is_compression_config_valid(config):
            raise ValueError("Invalid compression configuration!")
        
        model = copy.deepcopy(self).to(device=device)
        model.__dict__["_dmc"]["compression_config"] = config 
        model.__dict__["_dmc"]["input_shape"] = input_shape 
              
        for compression_type, compression_type_param in config.items():
            if compression_type == "prune_channel":
                
                if isinstance(config["prune_channel"]["sparsity"], (float, int)) and config["prune_channel"]["sparsity"] == 0:
                    continue

                def prune_channel_layer(layer): layer.is_pruned_channel = True

                # To prevent weight reselection especially in case of two step training
                # since the weight are not modified in place when pruning is done first
                # before QAT
                if self.is_pruned_channel and not force_prune_channel:
                    raise RuntimeError("Model has been pruned before to force for a repruning specify force_prune_channel as True")

                # Build masks first so weight_prune_channel exists before
                # is_pruned_channel=True is set — prevents AttributeError in
                # any forward pass that could fire between the two calls.
                model.init_prune_channel()
                model.apply(prune_channel_layer)

            elif compression_type == "quantize":
                def quantize_layer(layer): layer.is_quantized = True

                if config["quantize"]["scheme"] != QuantizationScheme.NONE:

                    # STRICT REQUIREMENT: Static Quantization requires Calibration Data.
                    if config["quantize"]["scheme"] == QuantizationScheme.STATIC and calibration_data is None:
                        raise ValueError(f"Pass a calibration data when doing static quantization!")

                    # Setting the quantization flag before quantization because it is needed for calibration 
                    model.apply(quantize_layer)
                    model.init_quantize(calibration_data)
            else:
                raise NotImplementedError(f"Compression of type {compression_type} not implemented!")

        return model
        


    def is_compression_config_valid(
        self, 
        compression_config:Dict[str, Any], 
        compression_keys:Optional[List]=None, 
        raise_error:bool=True
    ) -> bool:
        """
        Validates the compression configuration against DMC hardware constraints.

        This method ensures the requested compression parameters are feasible, checks
        the pruning and quantizations, if the are valid.

        Args:
            compression_config: Dictionary containing 'prune_channel' and 'quantize' settings.

        Returns:
            True if configuration is valid and deployable, False otherwise.
        """
        if compression_keys is None:
            compression_keys = list(compression_config.keys())

        for configuration_type in compression_keys:

            if configuration_type == "prune_channel":
                prune_channel_config = compression_config.get("prune_channel")

                sparsity = prune_channel_config.get("sparsity")

                # For uniform pruning
                if isinstance(sparsity, (float, int)):
                    if sparsity == 0:
                        continue
                    layer_sparsity = sparsity
                    sparsity = dict()
                    for name in self.names():
                        sparsity[name] = layer_sparsity
                # For non uniform pruning,
                elif isinstance(sparsity, dict):
                    def _all_leaves_numeric(d, path=""):
                        """Recursively verify every leaf in a nested sparsity dict is float/int."""
                        for k, v in d.items():
                            full = f"{path}.{k}" if path else k
                            if isinstance(v, dict):
                                ok = _all_leaves_numeric(v, full)
                                if not ok:
                                    return False
                            elif not isinstance(v, (float, int)):
                                if raise_error:
                                    raise TypeError(
                                        f"Sparsity leaf at '{full}' must be float or int, "
                                        f"got {type(v)}."
                                    )
                                return False
                        return True
                    for name, layer_sparsity in sparsity.items():
                        # Skip if layer cannot be pruned
                        if self[name].get_prune_channel_possible_hyperparameters() is None:
                            continue
                        if isinstance(layer_sparsity, dict):
                            if not _all_leaves_numeric(layer_sparsity, name):
                                return False
                        elif not isinstance(layer_sparsity, (float, int)):
                            if raise_error:
                                raise TypeError(
                                    f"Layer sparsity must be float, int, or dict (for Block/Branch per-sublayer control), "
                                    f"got {type(layer_sparsity)} for layer '{name}'."
                                )
                            return False
                        if name not in self.names():
                            if raise_error:
                                raise NameError(f"Found unknown layer name {name}")
                            return False
                        if isinstance(layer_sparsity, int):
                            hp = self[name].get_prune_channel_possible_hyperparameters()
                            # hp is a dict for Block/Branch; flatten to a combined range for validation
                            if isinstance(hp, dict):
                                def _collect_ranges(d):
                                    for v in d.values():
                                        if isinstance(v, dict):
                                            yield from _collect_ranges(v)
                                        else:
                                            yield v
                                all_ranges = list(_collect_ranges(hp))
                                valid = not all_ranges or layer_sparsity in range(min(len(r) for r in all_ranges))
                            else:
                                valid = hp is None or layer_sparsity in hp
                            if not valid:
                                if raise_error:
                                    raise ValueError(f"Received a layer_sparsity of {layer_sparsity} which is out of the valid range for layer '{name}'.")
                                return False
                    for name in self.names():
                        # if name not in sparsity and self.layers[name].is_prunable():
                        if name not in sparsity:
                            sparsity[name] = 0
                else:
                    if raise_error:
                        raise TypeError(f"prune sparsity has to be of type of float or dict not {type(sparsity)}!")
                    return False
                
                prune_channel_config["sparsity"] = sparsity

            elif configuration_type == "quantize":
                quantize_config = compression_config.get("quantize")
                scheme = quantize_config["scheme"]
                input_activation_bitwidth  = quantize_config["input_activation_bitwidth"]
                layer_activation_bitwidth  = quantize_config.get("layer_activation_bitwidth")
                granularity                = quantize_config["granularity"]
                layer_parameter_bitwidth   = quantize_config["layer_parameter_bitwidth"]

                if input_activation_bitwidth is not None and input_activation_bitwidth > 8:
                    if raise_error:
                        raise ValueError(f"Invalid activation bitwidth, {input_activation_bitwidth}")
                    return False

                # TODO: Confirm if this boolean expression is correct
                if (scheme == QuantizationScheme.NONE or scheme == QuantizationScheme.DYNAMIC) and input_activation_bitwidth is not None or \
                    input_activation_bitwidth is None and not (scheme == QuantizationScheme.NONE or scheme == QuantizationScheme.DYNAMIC):
                    if raise_error:
                        raise ValueError("When quantization scheme is NONE, activation bitwidth has to be None and vice versa.")
                    return False

                if scheme == QuantizationScheme.NONE: continue

                # ── layer_activation_bitwidth (STATIC only) ────────────────────
                # Only Conv2d and Linear create new output quantizers; all other
                # layers propagate the previous quantizer unchanged. The config
                # accepts either a uniform int or a per-layer dict (nested dicts
                # allowed for Block/Branch sub-layers). Missing entries default to 8.
                if scheme == QuantizationScheme.STATIC:
                    if layer_activation_bitwidth is None:
                        if raise_error:
                            raise ValueError("Static quantization requires layer_activation_bitwidth.")
                        return False

                    def _valid_ab_leaves(d, path):
                        for k, v in d.items():
                            full = f"{path}.{k}"
                            if isinstance(v, dict):
                                if not _valid_ab_leaves(v, full):
                                    return False
                            elif not isinstance(v, int) or v not in [2, 4, 8]:
                                if raise_error:
                                    raise TypeError(
                                        f"Nested layer_activation_bitwidth at '{full}' must be int in "
                                        f"[2, 4, 8], got {v}."
                                    )
                                return False
                        return True

                    def _layer_has_activation_bitwidth(layer):
                        hp = layer.get_quantize_possible_hyperparameters()
                        return hp is not None and "activation_bitwidth" in hp

                    if isinstance(layer_activation_bitwidth, int):
                        if layer_activation_bitwidth not in [2, 4, 8]:
                            if raise_error:
                                raise ValueError(
                                    f"Invalid layer_activation_bitwidth {layer_activation_bitwidth}; must be in [2, 4, 8]."
                                )
                            return False
                        uniform_ab = layer_activation_bitwidth
                        layer_activation_bitwidth = {
                            name: uniform_ab
                            for name in self.names()
                            if _layer_has_activation_bitwidth(self[name])
                        }
                    elif isinstance(layer_activation_bitwidth, dict):
                        for layer_name, ab in layer_activation_bitwidth.items():
                            if layer_name not in self.names():
                                if raise_error:
                                    raise NameError(f"Found unknown layer name '{layer_name}' in layer_activation_bitwidth.")
                                return False
                            if isinstance(ab, dict):
                                if not _valid_ab_leaves(ab, layer_name):
                                    return False
                            elif not isinstance(ab, int) or ab not in [2, 4, 8]:
                                if raise_error:
                                    raise ValueError(
                                        f"layer_activation_bitwidth for '{layer_name}' must be int in [2, 4, 8], got {ab!r}."
                                    )
                                return False
                        for name in self.names():
                            if name not in layer_activation_bitwidth and _layer_has_activation_bitwidth(self[name]):
                                layer_activation_bitwidth[name] = 8
                    else:
                        if raise_error:
                            raise TypeError(
                                f"layer_activation_bitwidth must be int or dict, got {type(layer_activation_bitwidth)}."
                            )
                        return False

                    quantize_config["layer_activation_bitwidth"] = layer_activation_bitwidth

                # ── layer_parameter_bitwidth / granularity ─────────────────────
                # For uniform quantization
                if isinstance(layer_parameter_bitwidth, int) and not isinstance(granularity, QuantizationGranularity) or \
                    isinstance(granularity, QuantizationGranularity) and not isinstance(layer_parameter_bitwidth, int):
                    if raise_error:
                        raise ValueError(f"When parameter bitwidth is a single value, granularity has to also be a single value and vice versa.")
                    return False

                # For non uniform quantization,
                if isinstance(layer_parameter_bitwidth, dict) and not isinstance(granularity, dict) or \
                    isinstance(granularity, dict) and not isinstance(layer_parameter_bitwidth, dict):
                    if raise_error:
                        raise ValueError(f"When parameter bitwidth is a dict, granularity has to also be a dict and vice versa.")
                    return False

                if isinstance(layer_parameter_bitwidth, int):
                    uniform_pb = layer_parameter_bitwidth
                    uniform_gran = granularity
                    layer_parameter_bitwidth = dict()
                    granularity = dict()
                    for name in self.names():
                        layer_parameter_bitwidth[name] = uniform_pb
                        granularity[name] = uniform_gran

                elif isinstance(layer_parameter_bitwidth, dict):
                    assert len(layer_parameter_bitwidth) == len(granularity) and layer_parameter_bitwidth.keys() == granularity.keys(), \
                            f"the keys of layer_parameter_bitwidth has to match with that of granularity"

                    def _valid_pb_leaves(d, path):
                        """Recursively validate every leaf is an int in [2, 4, 8]."""
                        for k, v in d.items():
                            full = f"{path}.{k}"
                            if isinstance(v, dict):
                                if not _valid_pb_leaves(v, full):
                                    return False
                            elif not isinstance(v, int) or v not in [2, 4, 8]:
                                if raise_error:
                                    raise TypeError(
                                        f"Nested layer_parameter_bitwidth at '{full}' must be int in [2, 4, 8], "
                                        f"got {v!r}."
                                    )
                                return False
                        return True

                    def _valid_gran_leaves(d, path):
                        """Recursively validate every leaf is a QuantizationGranularity."""
                        for k, v in d.items():
                            full = f"{path}.{k}"
                            if isinstance(v, dict):
                                if not _valid_gran_leaves(v, full):
                                    return False
                            elif not isinstance(v, QuantizationGranularity):
                                if raise_error:
                                    raise TypeError(
                                        f"Nested granularity at '{full}' must be QuantizationGranularity, "
                                        f"got {type(v)}."
                                    )
                                return False
                        return True

                    for (layer_name, pb), (_, gran) in zip(layer_parameter_bitwidth.items(), granularity.items()):
                        if layer_name not in self.names():
                            if raise_error:
                                raise NameError(f"Found unknown layer name {layer_name}")
                            return False
                        if isinstance(pb, dict):
                            if not _valid_pb_leaves(pb, layer_name):
                                return False
                            if not _valid_gran_leaves(gran, layer_name):
                                return False
                        else:
                            if not isinstance(pb, int):
                                if raise_error:
                                    raise TypeError(f"layer_parameter_bitwidth has to be int not {type(pb)} for layer {layer_name}!")
                                return False
                            if not isinstance(gran, QuantizationGranularity):
                                if raise_error:
                                    raise TypeError(f"granularity has to be QuantizationGranularity not {type(gran)} for layer {layer_name}!")
                                return False
                            if pb not in [2, 4, 8]:
                                if raise_error:
                                    raise ValueError(f"Received an invalid layer_parameter_bitwidth of {pb} for layer {layer_name}.")
                                return False
                    for name in self.names():
                        if name not in layer_parameter_bitwidth:
                            layer_parameter_bitwidth[name] = 8
                            granularity[name] = QuantizationGranularity.PER_TENSOR

                quantize_config["layer_parameter_bitwidth"] = layer_parameter_bitwidth
                quantize_config["granularity"] = granularity
                
            else:
                if raise_error:
                    raise ValueError(f"Invalid configuration scheme of {configuration_type}")                
                return False
        return True


    
    def init_prune_channel(self) -> None:
        """
        Executes Structured Channel Pruning.

        This method iterates through the network to identify and remove redundant 
        channels. Crucially, it handles **Dependency Propagation**: removing a 
        filter in layer `i` necessitates removing the corresponding input channel 
        in layer `i+1` to preserve connectivity and ensure the resulting model 
        remains dense.

        Side Effects:
            - Modifies layers in-place by attaching binary masks to weights.
            - Updates layer metadata to reflect reduced input/output shapes.
        """

        input_shape = self.__dict__["_dmc"]["input_shape"]
        sparsity = self.__dict__["_dmc"]["compression_config"]["prune_channel"]["sparsity"]
        metric = self.__dict__["_dmc"]["compression_config"]["prune_channel"]["metric"]

        keep_prev_channel_index = None

        # Prune all layers except last
        for name, layer in list(self.names_layers())[:-1]:

            keep_prev_channel_index = layer.init_prune_channel(
                sparsity[name], input_shape, keep_prev_channel_index, keep_current_channel_index=None,
                is_output_layer=False, metric=metric
            )
            input_shape = layer.get_output_tensor_shape(torch.Size(input_shape))

        # Prune last layer
        name, layer = list(self.names_layers())[-1]
        keep_prev_channel_index = layer.init_prune_channel(
            sparsity[name], input_shape, keep_prev_channel_index,keep_current_channel_index=None,
            is_output_layer=True, metric=metric
        )
        return 
    
    def get_prune_channel_possible_hyperparameters(self) -> Dict[str, Iterable]:
        """
        Defines the valid search space for Structured Pruning.

        This method queries every layer to determine how many channels can be 
        pruned while maintaining architectural validity. It is used to generate 
        the "Sparsity Sensitivity" graphs.

        Returns:
            Dictionary mapping layer names to their valid pruned channel counts.
            Note: The output layer is excluded ([:-1]) as its output dimensions 
            are fixed by the classification task (10 for MNIST, 100 for CIFAR).
        """
        prune_channel_possible_hypermeters = dict()

        def _expand_prune_hp(prefix, hp, out):
            if isinstance(hp, dict):
                for sub_name, sub_hp in hp.items():
                    _expand_prune_hp(f"{prefix}.{sub_name}", sub_hp, out)
            else:
                out[prefix] = hp

        for name, layer in list(self.names_layers())[:-1]:
            layer_prune_channel_possible_hypermeters = layer.get_prune_channel_possible_hyperparameters()
            if layer_prune_channel_possible_hypermeters is not None:
                _expand_prune_hp(f"sparsity.{name}", layer_prune_channel_possible_hypermeters, prune_channel_possible_hypermeters)

        # TODO: To extend to other metric type
        prune_channel_possible_hypermeters["metric"] = ["l2"]
        return prune_channel_possible_hypermeters
    
    def get_quantize_possible_hyperparameters(self, scheme:QuantizationScheme=QuantizationScheme.STATIC) -> Dict[str, Iterable]:
        """
        Defines the valid search space for Quantization.
        """
        if scheme != QuantizationScheme.STATIC:
            input_activation_bitwidth = [None]
        else:
            input_activation_bitwidth = [8, 4, 2]

        quantize_possible_hypermeters = dict()
        quantize_possible_hypermeters["scheme"] = [scheme]
        quantize_possible_hypermeters["input_activation_bitwidth"] = input_activation_bitwidth
        
        def _expand_quantize_hp(prefix, hp, out):
            if "parameter_bitwidth" in hp:
                out[f"parameter_bitwidth.{prefix}"] = hp["parameter_bitwidth"]
                out[f"granularity.{prefix}"] = hp["granularity"]
            if "activation_bitwidth" in hp:
                out[f"activation_bitwidth.{prefix}"] = hp["activation_bitwidth"]
            else:
                for sub_name, sub_hp in hp.items():
                    _expand_quantize_hp(f"{prefix}.{sub_name}", sub_hp, out)

        for name, layer in list(self.names_layers()):
            layer_quantize_possible_hypermeters = layer.get_quantize_possible_hyperparameters()
            if layer_quantize_possible_hypermeters is not None:
                _expand_quantize_hp(name, layer_quantize_possible_hypermeters, quantize_possible_hypermeters)

        return quantize_possible_hypermeters


    def get_compression_possible_hyperparameters(self, scheme:QuantizationScheme=QuantizationScheme.STATIC) -> Dict[str, Iterable]:
        """
        Defines the valid search space for Compression.
        
        :return: compression_hyperparameters: this is a dictionary with keys as
                 the compression parameter name from its root join by ".". Basically
                 it is a flatted dict with each level indicated by a "."
        """
        prune_hp = self.get_prune_channel_possible_hyperparameters()
        quant_hp = self.get_quantize_possible_hyperparameters(scheme)

        compression_hp = dict()
        for name, hp in prune_hp.items():
            compression_hp[f"prune_channel.{name}"] = hp
        
        for name, hp in quant_hp.items():
            compression_hp[f"quantize.{name}"] = hp

        return compression_hp
    
    def get_compression_possible_hyperparameters_size(self, scheme:QuantizationScheme=QuantizationScheme.STATIC) -> int:
            return math.prod(len(hyperparameters) for hyperparameters in self.get_compression_possible_hyperparameters(scheme).values())
    

    def get_compression_possible_hyperparameters_size_with_filter(self, scheme, filter):
        possible_configs = self.get_compression_possible_hyperparameters(scheme)
        config_keys = possible_configs.keys()
        config_values = possible_configs.values()
        
        count = 0
        for config in itertools.product(*config_values):
            config_dict = dict(zip(config_keys, config))
            
            if filter(config_dict):
                count += 1
        return count


    def decode_compression_dict_hyperparameter(self, compression_dict: Dict[str, Any]) -> Dict[str, Iterable]:
        """
        Reconstructs a hierarchical configuration dictionary from a flattened 
        search result.

        Args:
            compression_dict: Flat dict (e.g., {'quantize.bitwidth': 4})
        
        Returns:
            Nested dict (e.g., {'quantize': {'bitwidth': 4}}) suitable for `init_compress`.
        """
        compression_config = dict()
        for key, value in compression_dict.items():
            names = key.split(".")
            current_level = compression_config

            while len(names) > 0:
                current_name = names[0]
                if len(names) == 1:
                    current_level[current_name] = value
                else:
                    if current_name not in current_level:
                        current_level[current_name] = dict()
                    current_level = current_level[current_name]
                names.pop(0)

        return compression_config
        
    
    def init_quantize(self, calibration_data:Optional[torch.Tensor]=None) -> None:
        """
        Initializes the Quantization stage.

        This method configures the model for low-bitwidth inference. It can run in two modes,
        Dynamic Quantization and Static Quantization.

        Args:
            calibration_data: A single batch of representative input data. 
                              REQUIRED for Static Quantization to measure activation 
                              dynamic ranges (min/max) before training/deployment.
        """
        scheme = self.__dict__["_dmc"]["compression_config"]["quantize"]["scheme"]
        input_activation_bitwidth = self.__dict__["_dmc"]["compression_config"]["quantize"]["input_activation_bitwidth"]
        layer_activation_bitwidth = self.__dict__["_dmc"]["compression_config"]["quantize"]["layer_activation_bitwidth"]
        layer_parameter_bitwidth = self.__dict__["_dmc"]["compression_config"]["quantize"]["layer_parameter_bitwidth"]
        granularity = self.__dict__["_dmc"]["compression_config"]["quantize"]["granularity"]

        if scheme == QuantizationScheme.NONE:
            return

        elif scheme == QuantizationScheme.DYNAMIC:
            for name, layer in self.names_layers():
                layer.init_quantize(layer_parameter_bitwidth[name], granularity[name], scheme)
            return
        
        elif scheme == QuantizationScheme.STATIC:
            # We simulate this hardware constraint by placing a Quantize node at the very start of the network.
            setattr(self, "input_quantize", Quantize(
                self, input_activation_bitwidth, scheme, QuantizationGranularity.PER_TENSOR, scale_type=QuantizationScaleType.ASSYMMETRIC
            ))
            previous_output_quantize = self.input_quantize
            for name, layer in self.names_layers():
                previous_output_quantize = layer.init_quantize(
                    layer_parameter_bitwidth[name], granularity[name], 
                    scheme, layer_activation_bitwidth.get(name), previous_output_quantize,
                    current_output_quantize=None
                )

            # We run a forward pass with calibration data to let the observers
            self.train() # Ensure observers are updating
            if scheme == QuantizationScheme.STATIC:
                assert calibration_data is not None, f"Pass a calibration data when doing static quantization"
                self.to(calibration_data.device)
                self(calibration_data)

        return


    def fuse(self, batchnorm_only:bool=False, device:Optional[str]=None) -> "Sequential":
        """
        Fuses adjacent layers to optimize inference speed and reduce footprint.

        Operations performed:
        1. Conv2d + BatchNorm2d -> Fused Conv2d (folds BN parameters into weights).
        2. Conv2d + ReLU/ReLU6 -> Fused Conv2d (merges activation).
        3. Linear + ReLU/ReLU6 -> Fused Linear.

        This step is critical for the "Optimized Model for Inference" (Figure 1),
        reducing the number of distinct operations the microcontroller must execute.

        Args:
            batchnorm_only: If True, only folds BatchNorm layers (useful before QAT).
        
        Returns:
            A new Sequential model with fused layers.
        """


        # Helper to preserve DMC metadata (pruning masks/quantization config) 
        # when transferring to the new fused layer instance.
        def add_fused_layer(name, layer, fused_model, fused_layer=None):
            if fused_layer is not None:
                init_dmc_parameter(layer, fused_layer)
                fused_model.add_module(name, fused_layer) 
            else:
                fused_model.add_module(name, layer) 

        def fuse_container(container: Union[Sequential, Block, Branch]) -> Union[Sequential, Block, Branch]:

            if not isinstance(container, (Sequential, Block, Branch)):
                return container
                
            names_layers = list(container.names_layers())

            if isinstance(container, Sequential):
                fused_model = Sequential()
            elif isinstance(container, Block):
                fused_model = Block()
            elif isinstance(container, Branch):
                pass
            else:
                raise ValueError(f"Unsupported container type: {type(container)}")

            current_name, current_layer = names_layers[0]
            for next_name, next_layer in names_layers[1:]:
                is_fused = False

                if isinstance(current_layer, (Sequential, Block)):
                    current_layer = fuse_container(current_layer)

                elif isinstance(current_layer, Branch):
                    current_layer = Branch(
                        sublayer1=fuse_container(current_layer.sublayer1),
                        sublayer2=fuse_container(current_layer.sublayer2) if current_layer.sublayer2 is not None else None
                    )
                
                if isinstance(current_layer, Conv2d):
                    if isinstance(next_layer, BatchNorm2d):
                        fused_layer = fuse_conv2d_batchnorm2d(current_layer, next_layer)
                        add_fused_layer(current_name, current_layer, fused_model, fused_layer)
                        is_fused = True                     
                    elif isinstance(next_layer, ReLU) and not batchnorm_only:
                        fused_layer = fuse_conv2d_relu(current_layer, next_layer)
                        add_fused_layer(current_name, current_layer, fused_model, fused_layer)
                        is_fused = True                     
                    elif isinstance(next_layer, ReLU6) and not batchnorm_only:
                        fused_layer = fuse_conv2d_relu6(current_layer, next_layer)
                        add_fused_layer(current_name, current_layer, fused_model, fused_layer)
                        is_fused = True                     

                elif isinstance(current_layer, Linear):
                    if isinstance(next_layer, ReLU)  and not batchnorm_only:
                        fused_layer = fuse_linear_relu(current_layer, next_layer)
                        add_fused_layer(current_name, current_layer, fused_model, fused_layer)
                        is_fused = True                     
                    elif isinstance(next_layer, ReLU6) and not batchnorm_only:
                        fused_layer = fuse_linear_relu6(current_layer, next_layer)
                        add_fused_layer(current_name, current_layer, fused_model, fused_layer)
                        is_fused = True            

                # Update pointer for next iteration
                if is_fused:
                    current_layer = fused_layer
                else:
                    fused_model.add_module(current_name, current_layer)
                    current_layer = next_layer
                    current_name = next_name
                
            fused_model.add_module(current_name, current_layer)

            init_dmc_parameter(container, fused_model)
            
            return fused_model

        fused_model = fuse_container(self)
            
        if device:
            fused_model.to(device=device)

        return fused_model


    def get_size_in_bits(self) -> int:
        """Calculates total model size in bits (Sum of all packed layers)."""
        size = 0
        for layer in self.layers():
            size += layer.get_size_in_bits()
        return size
    

    def get_size_in_bytes(self) -> int:
        """
        Calculates total model binary size in Bytes.
        """
        return self.get_size_in_bits()//8



    def get_workspace_size(
        self, input_shape: Tuple,
        include_locals=False, include_runtime=False, ptr_size=2
    ) -> int:
        """
        Calculates the minimum SRAM (Static RAM) required for inference.

        This method estimates the "Ping-Pong" buffer sizes (Arena A and Arena B)
        needed for bare-metal deployment.

        Strategy:
        Instead of allocating memory for every intermediate tensor, DMC uses a
        shared buffers. The shared buffer is managed by either left aligning the data to the
        start or right aligning it to the end, ensuring no space fragmentation.
        Layer i reads its input from the start of the workspace and writes its output to be right
        aligned with the end of the workspace
        Layer i+1 reads its input from the end of the workspace with the none offset and writes its output
        to the start of the workspace

        :param input_shape: Dimensions of the network input.
        :param include_locals: Include buffer-read scalar locals held during layer computation.
        :param include_runtime: Include loop counters, fn pointers, and derived scalars.
        :param ptr_size: Pointer byte width (default 2 for AVR).

        :return: max_layer_acitivation_workspace_size: The peak byte requirements for the activations of the model.
        """

        if isinstance(input_shape, tuple):
            input_shape = torch.Size(input_shape)
        # data_per_byte = get_data_bits(self)

        data_bits = get_data_bits(self)
        
        max_layer_workspace_size = pad_bits_to_byte(input_shape.numel() * data_bits)
        output_shape = input_shape
        for layer in self.layers():
            layer_workspace_size = layer.get_workspace_size(
                torch.Size(output_shape), include_locals, include_runtime, ptr_size
            )
            output_shape = layer.get_output_tensor_shape(torch.Size(output_shape))
            max_layer_workspace_size = max(max_layer_workspace_size, layer_workspace_size)

        if not (include_locals or include_runtime):
            return max_layer_workspace_size

        def _count_all_layers(module) -> int:
            """Recursively count every registered sub-module (Layer instance) in the tree."""
            count = 0
            for child in module._modules.values():
                count += 1
                count += _count_all_layers(child)
            return count

        total_layers = _count_all_layers(self)
        scheme = None
        if self.is_quantized:
            scheme = self.__dict__["_dmc"]["compression_config"]["quantize"]["scheme"]
        # Buffer-read scalars: uint32_t workspace_size (4) + uint8_t layers_len (1) [+ uint8_t quantize_property (1) for SQ]
        buf_locals = 5 + (1 if scheme == QuantizationScheme.STATIC else 0)
        # All Layer objects anywhere in the tree live in static RAM (ptr_size each for buffer ptr).
        seq_locals  = (buf_locals + total_layers * ptr_size) if include_locals else 0
        # uint8_t i loop counter + Layer* temp for indirect call
        seq_runtime = (1 + ptr_size) if include_runtime else 0

        return max_layer_workspace_size + seq_locals + seq_runtime


    @torch.no_grad()
    def convert_to_c(
        self, 
        input_shape:Tuple, 
        var_name:str, 
        src_dir:str="./", 
        include_dir:str = "./", 
        for_arduino:bool=False,
        test_input:Optional[torch.Tensor]=None
    ) -> None:
        """Generate C code for deployment
        
        Args:
            var_name: Base name for generated files
            dir: Output directory for generated files
            input_shape: Shape of the model input tensor
            test_input: Optional test input tensor to generate C array
            for_arduino: If True, generates Arduino-compatible C code, add PROGMEM if needed to ensure the params are stored in flash memory.
        """
        """
        Generates the dependency-free C library for bare-metal deployment.

        This method corresponds to the "Encode Model Parameters" -> "Output Continuous 
        Byte Stream".

        It produces three artifacts:
        1.  {var_name}.h: Header file defining the model structure and workspace.
        2.  {var_name}_def.cpp: The model definition linking layers together.
        3.  {var_name}_params.cpp: The heavy weight data (Bit-Packed arrays).

        Key DMC Features Implemented:
        - Static Allocation: Calculates `MAX_OUTPUT_EVEN/ODD_SIZE` to define 
          fixed `int8_t workspace[]` arrays, preventing runtime malloc calls.
        - Integer Types: If Static Quantization is active, generates `int8_t` 
          interfaces; otherwise falls back to `float`.
        """
        def write_str_to_c_file(file_str: str, file_name: str, dir: str):
            """Helper to write string to file"""
            with open(os.path.join(dir, file_name), "w") as file:
                file.write(file_str)
        
        # Initialize file contents
        header_file = f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n"
        if not for_arduino:
            header_file += "#include <stdint.h>\n#include \"deep_microcompression.h\"\n\n\n"
        else:
            header_file += "#include <stdint.h>\n#include <Arduino.h>\n#include \"deep_microcompression.h\"\n\n\n"

        definition_file = f"#include \"{var_name}.h\"\n\n"
        param_definition_file = f"#include \"{var_name}.h\"\n\n"
    
        input_shape = torch.Size(input_shape)
        original_input_shape = input_shape  # saved for test_input HWC permutation
        # Calculate workspace requirements
        max_layer_acitivation_workspace_size = self.get_workspace_size(torch.Size(input_shape))
        # max_output_even_size, max_output_odd_size = self.get_max_workspace_arena(input_shape)
        
        scheme = None
        if self.is_quantized:
            scheme = self.__dict__["_dmc"]["compression_config"]["quantize"]["scheme"]

        progmem = "PROGMEM " if for_arduino else ""
        # get_workspace_size() always returns bytes.
        # The C array is typed (float or int8_t), so WORKSPACE_SIZE must be in
        # *elements* of that type, not bytes.
        ws_bytes = max_layer_acitivation_workspace_size

        if scheme != QuantizationScheme.STATIC:
            ws = ws_bytes // 4   # float32: 4 bytes per element
            workspace_type   = FLOAT_T
            class_suffix     = ""
            seq_params_info  = [
                (UINT32_T, "workspace_size", str(ws)),
                (UINT8_T,  "layers_len",     str(len(self))),
            ]
        else:
            ws = ws_bytes        # int8_t: 1 byte per element
            input_activation_bitwidth = self.__dict__["_dmc"]["compression_config"]["quantize"]["input_activation_bitwidth"]
            if input_activation_bitwidth == 8:
                quantize_property = ACTIVATION_BITWIDTH_8
            elif input_activation_bitwidth == 4:
                quantize_property = ACTIVATION_BITWIDTH_4
            elif input_activation_bitwidth == 2:
                quantize_property = ACTIVATION_BITWIDTH_2
            else:
                raise QuantizationBitWidthError(input_activation_bitwidth)
            workspace_type   = INT8_T
            class_suffix     = "_SQ"
            seq_params_info  = [
                (UINT32_T, "workspace_size",     str(ws)),
                (UINT8_T,  "layers_len",         str(len(self))),
                (UINT8_T,  "quantize_property",  quantize_property),
            ]

        workspace_header = (
            f"#define WORKSPACE_SIZE {ws}\n"
            f"extern {workspace_type} workspace[WORKSPACE_SIZE];\n\n"
        )
        workspace_def = f"{workspace_type} workspace[WORKSPACE_SIZE];\n\n"

        # Generate per-layer code and collect layer pointers for the Sequential buffer
        layer_header_acc = ""
        pre_flatten_shape = None  # shape going into the most recent Flatten, for Linear weight reordering

        for layer_name, layer in self.names_layers():
            seq_params_info.append((VOID_PTR, layer_name, f"(void*)&{layer_name}"))

            if isinstance(layer, Linear):
                layer_header, layer_def, layer_param_def = layer.convert_to_c(
                    layer_name, input_shape, for_arduino=for_arduino, conv_shape=pre_flatten_shape
                )
                pre_flatten_shape = None
            else:
                layer_header, layer_def, layer_param_def = layer.convert_to_c(
                    layer_name, input_shape, for_arduino=for_arduino
                )
                pre_flatten_shape = input_shape if isinstance(layer, Flatten) else None

            layer_header_acc    += layer_header
            param_definition_file += layer_param_def
            definition_file     += layer_def
            input_shape = layer.get_output_tensor_shape(torch.Size(input_shape))

        # Build Sequential packed-struct buffer (all pointers in Flash)
        fields  = ''.join([f'    {t} {n};\n' for t, n, _ in seq_params_info])
        values  = ',\n    '.join([v for _, _, v in seq_params_info])
        seq_buf = (
            f"static const struct __attribute__((packed)) {{\n"
            f"{fields}"
            f"}} {var_name}_buffer {progmem}= {{\n"
            f"    {values}\n"
            f"}};\n"
            f"Sequential{class_suffix} {var_name}((const uint8_t*)&{var_name}_buffer, workspace, WORKSPACE_SIZE);\n"
        )

        layers_header = (
            f"#define LAYERS_LEN {len(self)}\n"
            f"extern Sequential{class_suffix} {var_name};\n\n"
        )

        header_file  += workspace_header + layers_header + layer_header_acc
        header_file  += f"\n#endif //{var_name.upper()}_h\n"
        definition_file += workspace_def + seq_buf

        # Write files
        write_str_to_c_file(header_file, f"{var_name}.h", include_dir)
        write_str_to_c_file(definition_file, f"{var_name}_def.cpp", src_dir)
        write_str_to_c_file(param_definition_file, f"{var_name}_params.cpp", src_dir)


        if test_input is not None:
            # Permute test_input from PyTorch CHW/NCHW layout to HWC for deployment
            if len(original_input_shape) == 3:
                if test_input.dim() == 4 and test_input.shape[0] == 1:
                    test_input = test_input.squeeze(0).permute(1, 2, 0).contiguous()
                elif test_input.dim() == 3:
                    test_input = test_input.permute(1, 2, 0).contiguous()

            if self.is_quantized and self.__dict__["_dmc"]["compression_config"]["quantize"]["scheme"] == QuantizationScheme.STATIC:
                _, test_input_def = convert_tensor_to_bytes_var(
                    self.input_quantize.apply(test_input), 
                    "test_input", 
                    self.input_quantize.bitwidth,
                    for_arduino=for_arduino
                )
                # test_input_def = f"\nconst int8_t test_input[] = {{\n"
                # for line in torch.split(self.input_quantize.apply(test_input).flatten(), 28):
                #     test_input_def += "    " + ", ".join(
                #         [f"{val:4d}" for val in line]
                #     ) + ",\n"
                # test_input_def += "};\n"
            else:
                    _, test_input_def = convert_tensor_to_bytes_var(test_input, "test_input",
                for_arduino=for_arduino)
                    if not for_arduino:
                        test_input_def = f"\nconst float test_input[] = {{\n"
                    else:
                        test_input_def = f"#include <Arduino.h>\n\nconst float test_input[] PROGMEM= {{\n"

                    for line in torch.split(test_input.flatten(), 28):
                        test_input_def += "    " + ", ".join(
                            [f"{val:.4f}" for val in line]
                        ) + ",\n"
                    test_input_def += "};\n"
            write_str_to_c_file(test_input_def, f"{var_name}_test_input.h", include_dir)

        return


    def get_layers_prune_channel_sensity(
        self, 
        input_shape, 
        data_loader, 
        metrics, 
        device="cpu",
        train = False,
        train_dataloader = None,
        epochs = None,
        criterion_fun = None,
        optimizer_fun = None,
        lr_scheduler = None,
        callbacks = [],
    ) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    
        """
        Performs Layer-wise Sensitivity Analysis.

        This method isolates each layer and prunes it at varying sparsity levels 
        while keeping other layers dense. It quantifies the "collapse point" where 
        accuracy degrades significantly.

        Args:
            train: If True, performs "Retraining" (Fine-tuning) after pruning to 
                   measure accuracy recovery (Fig 3b vs 3a).
        
        Returns:
            Nested Dictionary: {metric -> {layer_name -> [(sparsity_ratio, acc_before, acc_after)]}}
        """
        if train:
            assert train_dataloader is not None
            assert epochs is not None
            assert criterion_fun is not None
            assert optimizer_fun is not None

        prune_channel_hp = self.get_prune_channel_possible_hyperparameters()
        prune_channel_layers_sensity = {metric_name: dict() for metric_name in metrics.keys()}

        for layer_name, layer_prune_channel_hp in tqdm(prune_channel_hp.items()):

            for metric_name in metrics.keys():
                prune_channel_layers_sensity[metric_name].update({layer_name : list()})

            max_layer_prune_channel_hp = layer_prune_channel_hp.stop
            for layer_prune_channel in layer_prune_channel_hp:
                compression_config = {
                    "prune_channel" :{
                        "sparsity" : {
                            layer_name: layer_prune_channel
                        },
                        "metric" : "l2"
                    },
                }
                prune_channel_model = self.init_compress(config=compression_config, input_shape=input_shape)
                prune_channel_model_metrics = prune_channel_model.evaluate(data_loader=data_loader, metrics=metrics, device=device)

                if train:
                    optimizer_fun = torch.optim.SGD(prune_channel_model.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4)
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fun, mode="min", patience=1)

                    prune_channel_model.fit(
                        train_dataloader=train_dataloader,
                        epochs=epochs,
                        criterion_fun=criterion_fun,
                        optimizer_fun=optimizer_fun,
                        lr_scheduler=lr_scheduler,
                        validation_dataloader=data_loader,
                        metrics=metrics,
                        verbose=False,
                        callbacks=callbacks,
                        device=device
                    )

                    prune_channel_model_metrics_train = prune_channel_model.evaluate(data_loader=data_loader, metrics=metrics, device=device)

                    for metric_name in metrics.keys():
                        prune_channel_layers_sensity[metric_name][layer_name].append((layer_prune_channel/max_layer_prune_channel_hp, prune_channel_model_metrics[metric_name], prune_channel_model_metrics_train[metric_name]))
                else:
                    for metric_name in metrics.keys():
                        prune_channel_layers_sensity[metric_name][layer_name].append((layer_prune_channel/max_layer_prune_channel_hp, prune_channel_model_metrics[metric_name]))
        return prune_channel_layers_sensity
    

    def get_weight_distributions(self, bins=256) -> Dict[str, Optional[torch.Tensor]]:
        """Get weight histograms for all layers
        
        Args:
            bins: Number of histogram bins
            
        Returns:
            Dictionary mapping layer names to weight histograms
        """
        weight_dist = dict()
        for name, layer in self.names_layers():
            if hasattr(layer, "weight"): 
                weight_dist[name] = torch.histogram(layer.weight.cpu(), bins=bins)
            else: 
                weight_dist[name] = None
        return weight_dist
    
