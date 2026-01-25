from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Any

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

        # Flag: (Structured Pruning) is active
        self.is_pruned_channel = False
        
        # Flag: (Quantization) is active
        self.is_quantized = False

    @property
    def is_compressed(self):
        """Returns True if any optimization pass has been applied."""
        return self.is_pruned_channel or self.is_quantized
    
    @abstractmethod
    def init_prune_channel(
        self, 
        sparsity: float, 
        keep_prev_channel_index: Optional[torch.Tensor], 
        input_shape: torch.Size,
        is_output_layer: bool = False, 
        metric: str = "l2"
    ) -> Optional[torch.Tensor]:
        """
        Structured Pruning Setup.
        
        Must implement the logic to:
        1. Calculate filter importance (e.g., L1/L2 norm).
        2. Generate binary masks for weights.
        3. Handle dependency propagation (return indices of kept channels).
        """
        pass
    
    @abstractmethod
    def get_prune_channel_possible_hyperparameters(self):
        """
        Returns the valid range of channels that can be kept (for Search/NAS).
        Used to generate the Sensitivity Analysis graphs.
        """
        pass


    @abstractmethod
    def init_quantize(
        self, 
        parameter_bitwidth: int, 
        granularity: QuantizationGranularity, 
        scheme: QuantizationScheme,
        activation_bitwidth:Optional[int]=None,
        previous_output_quantize: Optional[Quantize] = None
    ):
        """
        Quantization Setup.
        
        Must implement the logic to:
        1. Attach Input/Weight/Output quantization observers.
        2. Propagate scale factors from the previous layer (for static inference).
        """
        if "quantize" not in self.__dict__["_dmc"]:
            self.__dict__["_dmc"]["quantize"] = dict()
        self.__dict__["_dmc"]["quantize"]["scheme"] = scheme
        self.__dict__["_dmc"]["quantize"]["granularity"] = granularity
        self.__dict__["_dmc"]["quantize"]["parameter_bitwidth"] = parameter_bitwidth
        self.__dict__["_dmc"]["quantize"]["activation_bitwidth"] = activation_bitwidth

        pass


    @abstractmethod
    def get_quantize_possible_hyperparameters(self):
        return {
            "parameter_bitwidth": [8, 4, 2], 
            "granularity": [QuantizationGranularity.PER_TENSOR, QuantizationGranularity.PER_CHANNEL]
        }
    

    @abstractmethod
    def get_compression_parameters(self):
        """
        Retrieves the final compressed parameters (Weights/Biases).
        
        This method must apply all active pruning masks and simulation steps 
        to return the exact tensors that will be exported to C.
        """
        pass


    @abstractmethod
    def get_size_in_bits(self) -> int:
        """Calculates the theoretical size of the layer in bits."""
        pass

    def get_size_in_bytes(self):
        return self.get_size_in_bits() // 8
    
    def get_size_in_KB(self):
        return self.get_size_in_bits() / (8 * 1024)


    @abstractmethod
    def get_output_tensor_shape(self, input_shape):
        """
        Calculates output dimensions.
        
        Crucial for the "Ping-Pong" SRAM estimation (`get_max_workspace_arena`).
        Must return: (Max_Intermediate_Shape, Final_Output_Shape).
        """
        pass

    @abstractmethod
    def convert_to_c(self, var_name, input_shape, for_arduino=False):
        """
        Deployment Generation.

        Args:

            var_name: Variable name to use in generated code
            input_shape: Shape of the input tensor
            for_arduino: Flag for Arduino-specific code generation, to add PROGMEM if needed
        
        Must implement:
        1. Bit-Packing for weights.
        2. C struct definition generation.
        3. Parameter array hex dumping.
        """
        pass



