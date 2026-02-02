from typing import Dict, Iterable, List, Any, Union

import torch


class ConfigEncoder:
    """
    Handles the conversion of a raw configuration dictionary into a 
    flat Tensor suitable for the neural network.
    """
    def __init__(
            self, 
            model,
            categorical_keys:List[str] = [
                "prune_channel.metric", 
                "quantize.scheme", 
                "quantize.activation_bitwidth",
                "quantize.parameter_bitwidth",
                "quantize.granularity"
            ],
            dtype:torch.dtype=torch.float32,
            device:torch.device=torch.device("cpu")
        ):
        self.search_space = model.get_compression_possible_hyperparameters()
        self.encoders = {}

        # Categorical/Enum Fields (One-Hot Encoding)
        self.categorical_keys = categorical_keys
        self.dtype = dtype
        self.device = device
        
        # Pre-compute normalisation constants and category maps
        self._setup_encoders()

    def _setup_encoders(self):
        """
        Sets up the encoder to handle the data normalisation and one-hot encoding for categorical mapping
        """
        # Normalize Integer Ranges, with the max value
        for key in self.search_space:
            # removing the layer name
            if key not in self.categorical_keys and ".".join(key.split(".")[:-1]) not in self.categorical_keys:
                # Store the max value for normalization (e.g., 6, 16, 84)
                self.encoders[key] = {"type": "nominal", "max": max(self.search_space[key])}
            else:
                # Create a mapping from Value -> Index
                # Handle 'None' explicitly by converting everything to string for consistency
                unique_vals = list(self.search_space[key])
                val_to_idx = {str(v): i for i, v in enumerate(unique_vals)}
                
                self.encoders[key] = {
                    "type": "categorical", 
                    "map": val_to_idx, 
                    "size": len(unique_vals)
                }


    def encode(self, config:Dict[str, Union[Any, Iterable]], with_metric:bool=False, metric_key:str="metric", normalise_norminal:bool=False) -> torch.Tensor:
        """
        Encodes the network configuration to numerical types that can be ingested by the estimator
        model 

        :param: config: the model congiration to be converted to a valid input for the estimator
        :param: with_metric: if the config has metric item in it, it seperate when you are trying to
                for training vs prediction

        :return: config_tensor: the tensor of the model config encoded to a numerical format to be
                 be ingested by the model
        """
        vector = []
        
        # if the values are list, meaning it not just a single point as can be provided by estimator
        length = None
        for i, (config_key, config_value) in enumerate(config.items()):
            if isinstance(config_value, list):
                if i == 0: length = len(config_value)
                else:
                    assert len(config_value) == length, (
                        f"if any configuration value is a list then all "
                        f"should be of the same length, got {config_key} to be of length "
                        f"{len(config)} and working with length of {length}"
                    )

        if length is None:
            # Encode integer type, normalized as float
            for key in self.search_space:
                # removing the layer name
                if key not in self.categorical_keys and ".".join(key.split(".")[:-1]) not in self.categorical_keys:
                    val = config.get(key)
                    if normalise_norminal:
                        val /= self.encoders[key]["max"]
                    vector.append(val)
                else:
                    # Encode Categorical (One-Hot)
                    enc_info = self.encoders[key]
                    val = str(config.get(key)) # Convert input to string to match key map
                    
                    # Create One-Hot Vector
                    one_hot = [0] * enc_info['size']
                    if val in enc_info['map']:
                        idx = enc_info['map'][val]
                        one_hot[idx] = 1
                    vector.extend(one_hot)
            # getting the metric
            if with_metric:
                vector.append(config.get(metric_key))
        
        else:
            for i in range(length):
                vector_i = []
                # Encode integer type, normalized as float
                for key in self.search_space:
                    # removing the layer name
                    if key not in self.categorical_keys and ".".join(key.split(".")[:-1]) not in self.categorical_keys:
                        val = config.get(key)[i]
                        vector_i.append(val)

                    else:
                        # Encode Categorical (One-Hot)
                        enc_info = self.encoders[key]
                        val = str(config.get(key)[i]) # Convert input to string to match key map
                        if normalise_norminal:
                            val /= self.encoders[key]["max"]
                        
                        # Create One-Hot Vector
                        one_hot = [0] * enc_info['size']
                        if val in enc_info['map']:
                            idx = enc_info['map'][val]
                            one_hot[idx] = 1
                        vector_i.extend(one_hot)
                # getting the metric
                if with_metric:
                    vector_i.append(config.get(metric_key)[i])
                
                vector.append(vector_i)
        return torch.tensor(vector, dtype=self.dtype, device=self.device)


    def __call__(self, config:Dict[str, Union[Any, Iterable]], with_metric:bool=False) -> torch.Tensor:
        """
        Encodes a model compression configuration
        """
        return self.encode(config, with_metric)
    