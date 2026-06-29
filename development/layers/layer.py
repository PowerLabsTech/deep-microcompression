import warnings
from abc import ABC, abstractmethod
from typing import Optional

from ..compressors import QuantizationScheme, QuantizationGranularity, Quantize
import torch


class Layer(ABC):
    """
    Interface for quantization-aware and pruning-aware layers.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initializes DMC state flags.
        
        All DMC layers track their own compression state to handle the 
        transition from Float32 -> Pruned -> Quantized.
        """
        setattr(self, "_dmc", dict())
        super().__init__(*args, **kwargs)
        self.is_pruned_channel = False
        self.is_quantized = False

    @property
    def is_compressed(self):
        """Returns True if any optimization pass has been applied."""
        return self.is_pruned_channel or self.is_quantized

    # ------------------------------------------------------------------
    # Pruning interface
    # ------------------------------------------------------------------

    def init_prune_channel(
        self,
        sparsity: float,
        input_shape: torch.Size,
        keep_prev_channel_index: Optional[torch.Tensor],
        keep_current_channel_index: Optional[torch.Tensor],
        is_output_layer: bool = False,
        metric: str = "l2",
    ) -> Optional[torch.Tensor]:
        """
        Default: passthrough — layers with no learnable weights just forward indices.
        Learnable layers (Conv2d, Linear) override this to compute importance and build masks.
        """
        if keep_current_channel_index is None:
            return keep_prev_channel_index
        return keep_current_channel_index

    def get_prune_channel_possible_hyperparameters(self):
        """
        Returns valid pruning search space, or None if the layer is not prunable.
        Overridden by Conv2d and Linear to return range(out_channels/out_features).
        """
        return None

    # ------------------------------------------------------------------
    # Quantization interface
    # ------------------------------------------------------------------

    def init_quantize(
        self,
        parameter_bitwidth,
        granularity,
        scheme,
        activation_bitwidth=None,
        previous_output_quantize=None,
        current_output_quantize: Optional[Quantize] = None,
        change_quantization_scale:bool = False
    ):
        """
        Stores quantization config in _dmc and propagates the output quantizer.

        Default (passthrough): returns current_output_quantize if provided, otherwise
        previous_output_quantize. Learnable layers (Conv2d, Linear) override this to
        attach weight/input/output quantizers and return a new output quantizer.
        """
        if activation_bitwidth is None and scheme == QuantizationScheme.STATIC and previous_output_quantize is not None:
            activation_bitwidth = previous_output_quantize.bitwidth

        if not change_quantization_scale:
            if previous_output_quantize is not None and activation_bitwidth is not None:
                assert previous_output_quantize.bitwidth == activation_bitwidth, (
                    f"Recieved a activition bitwidth different from its input bitwidth for"
                    f" a none quantization scale changing layer, {self.__class__.__name__}."
                )

        if "quantize" not in self.__dict__["_dmc"]:
            self.__dict__["_dmc"]["quantize"] = dict()
        self.__dict__["_dmc"]["quantize"]["scheme"] = scheme
        self.__dict__["_dmc"]["quantize"]["granularity"] = granularity
        self.__dict__["_dmc"]["quantize"]["parameter_bitwidth"] = parameter_bitwidth
        self.__dict__["_dmc"]["quantize"]["activation_bitwidth"] = activation_bitwidth
        if previous_output_quantize is not None:
            self.__dict__["_dmc"]["quantize"]["input_activation_bitwidth"] = previous_output_quantize.bitwidth
        else:
            self.__dict__["_dmc"]["quantize"]["input_activation_bitwidth"] = None

        if current_output_quantize is None:
            return previous_output_quantize
        warnings.warn(
            f"{self} received a current_output_quantize, forcing it to use that as the "
            "quantization base instead of the previous layer's quantizer. This is expected "
            "when the layer sits inside a Branch alongside a modifying layer (Conv/Linear).",
            UserWarning,
            stacklevel=2,
        )
        return current_output_quantize

    def get_quantize_possible_hyperparameters(self):
        """
        Returns valid quantization search space, or None if the layer has no parameters.
        Overridden by Conv2d and Linear to return bitwidth/granularity options.
        """
        return None

    # ------------------------------------------------------------------
    # Export / sizing interface (all abstract — every layer must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_compression_parameters(self):
        """Returns final hard-pruned and hard-quantized tensors for C export."""
        pass

    @abstractmethod
    def get_workspace_size(
        self, input_shape, include_locals=False,
        include_runtime=False, ptr_size=2
    ) -> int:
        return 0

    @abstractmethod
    def get_size_in_bits(self) -> int:
        """Calculates the theoretical compressed size of the layer in bits."""
        pass

    def get_size_in_bytes(self):
        return self.get_size_in_bits() // 8

    def get_size_in_KB(self):
        return self.get_size_in_bits() / (8 * 1024)

    @abstractmethod
    def get_output_tensor_shape(self, input_shape):
        """Calculates output dimensions — used for Ping-Pong SRAM estimation."""
        pass

    @abstractmethod
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """Generates C header, definition, and parameter strings for bare-metal deployment."""
        pass

    def get_packed_struct(self, var_name, params_info, for_arduino=False):
        """Generates a named packed Flash struct with no class instantiation."""
        progmem = "PROGMEM " if for_arduino else ""
        return (
            f"static const struct __attribute__((packed)) {{\n"
            f"{''.join([f'    {t} {n};\n' for t, n, _ in params_info])}"
            f"}} {var_name} {progmem}= {{\n"
            f"    {', '.join([v for _, _, v in params_info])}\n"
            f"}};\n"
        )

    def get_struct_def(self, var_name, params_info, quantize_scheme=QuantizationScheme.NONE, for_arduino=False):
        suffix = "" if quantize_scheme == QuantizationScheme.NONE else \
                 "_DQ" if quantize_scheme == QuantizationScheme.DYNAMIC else "_SQ"
        return (
            self.get_packed_struct(f"{var_name}_buffer", params_info, for_arduino)
            + f"{self.__class__.__name__}{suffix} {var_name}((uint8_t*)&{var_name}_buffer);\n"
        )
    