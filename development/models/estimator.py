"""
@file estimator.py
@brief Neural Network Estimator for Hyperparameter Search.

This module defines a lightweight Multilayer Perceptron (MLP) used to approximate
the relationship between compression parameters (e.g., sparsity configurations) 
and model performance. It serves as a surrogate model to speed up the search 
process for optimal compression ratios.
"""
from typing import Dict, Iterable, List, Any, Union

import torch
import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .sequential import Sequential

class Estimator:
    """
    A regression model used to predict the performance metric of a compressed model.
    
    This class wraps a custom MLP constructed using DMC's `Sequential` container.
    It includes built-in data normalization, training (fit), and inference (predict)
    routines tailored for the architecture search phase.
    """
    
    def __init__(
        self, 
        data: Dict[str, List], 
        search_space:Dict,
        hidden_dim:List[int] = [64, 128, 64], 
        dropout:float = 0.5, 
        device:str = "cpu"
    ) -> None:
        """
        Initialize the Estimator model.

        Args:
            data: Input dataset used to initialize normalization statistics and 
                  determine input feature dimensions. Expected shape: (N, features + target).
            hidden_dim: List of integers defining the size of hidden layers.
            dropout: Dropout probability for regularization.
            device: Computation device ('cpu' or 'cuda').
        """
        self.device = device

        assert len(list(data.values())[0]) > 3, f"The data must have at least three datapoints got {len(list(data.values())[0])}"
        self.encoder = ConfigEncoder(search_space=search_space)
        self.data = torch.tensor(self.encoder(data, with_metric=True), dtype=torch.float32, device=self.device)

        self.x_mu = torch.tensor([0.0], device=device)
        self.x_std = torch.tensor([1.0], device=device)

        layers = []
        input_features = self.data.size(1) - 1

        layers.append(nn.Linear(input_features, hidden_dim[0]))
        layers.append(nn.BatchNorm1d(hidden_dim[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        input_features = hidden_dim[0]
        for h in hidden_dim[1:]:
            output_features = h

            layers.append(nn.Linear(input_features, output_features))
            layers.append(nn.BatchNorm1d(output_features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

            input_features = output_features
        layers.append(nn.Linear(input_features, 1))
        self.model = Sequential(*layers).to(self.device)

        self.optimizer_fun = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion_fun = nn.MSELoss()


    def fit(self, epochs, batch_size=32):
        """
        Trains the estimator on the provided data.
        """
        self.model.train()
        X, Y = self.data[:, :-1], self.data[:, -1].unsqueeze(dim=1)
        
        self.x_mu = X.mean(dim=0).to(self.device)
        x_std = X.std(dim=0)
        self.x_std = torch.where(x_std > 1e-10, x_std, torch.ones_like(x_std)).to(self.device)

        N = X.size(0)
        assert N > 1, "The number of data point is less than 2, it can not be split."
        idx = torch.randperm(N, device=self.device)

        n_val = max(1, int(N * 0.2))  # at least 1 val sample if N > 1
        n_val = min(n_val, N - 1)     # but leave at least 1 for training

        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64, shuffle=False)

        history = self.model.fit(
            train_dataloader=train_loader, 
            epochs=epochs, 
            criterion_fun=self.criterion_fun, 
            optimizer_fun=self.optimizer_fun, 
            validation_dataloader=val_loader,
            metrics={
                "mse": lambda y_pred, y_true: torch.pow((y_pred - y_true), 2).mean().item(), 
                "rmse": lambda y_pred, y_true: torch.sqrt(torch.pow((y_pred - y_true), 2).mean()).item(),
                "abs": lambda y_pred, y_true: (y_pred- y_true).abs().mean().item(),
            },
            device=self.device, 
            batch_size=batch_size
        )
        return history
    
    @torch.no_grad()
    def predict(self, config:Dict[str, Any]):
        """
        Predicts the metric for a given input configuration.
        """
        X = self.encoder(config, with_metric=False)
        self.model.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)

        if X.dim() == 1:
            X = X.unsqueeze(0)
        return self.model(X).item()


class ConfigEncoder:
    """
    Handles the conversion of a raw configuration dictionary into a 
    flat Tensor suitable for the neural network.
    """
    def __init__(
            self, 
            search_space: Dict[str, Iterable],
            categorical_keys:List[str] = [
                "prune_channel.metric", 
                "quantize.scheme", 
                "quantize.granularity", 
                "quantize.bitwidth"
            ],
            dtype:torch.dtype=torch.float32
        ):
        self.search_space = search_space
        self.encoders = {}
        self.feature_dim = 0

        # Categorical/Enum Fields (One-Hot Encoding)
        self.categorical_keys = categorical_keys
        self.dtype = dtype
        
        # Pre-compute normalisation constants and category maps
        self._setup_encoders()

    def _setup_encoders(self):
        """
        Sets up the encoder to handle the data normalisation and one-hot encoding for categorical mapping
        """
        # Normalize Integer Ranges, with the max value
        for key in self.search_space:
            if key not in self.categorical_keys:
                # Store the max value for normalization (e.g., 6, 16, 84)
                self.encoders[key] = {"type": "nominal", "max": max(self.search_space[key])}
                self.feature_dim += 1

        for key in self.categorical_keys:
            # Create a mapping from Value -> Index
            # Handle 'None' explicitly by converting everything to string for consistency
            unique_vals = list(self.search_space[key])
            val_to_idx = {str(v): i for i, v in enumerate(unique_vals)}
            
            self.encoders[key] = {
                "type": "categorical", 
                "map": val_to_idx, 
                "size": len(unique_vals)
            }
            self.feature_dim += len(unique_vals)


    def encode(self, config:Dict[str, Union[Any, Iterable]], with_metric:bool=False) -> torch.Tensor:
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
                if key not in self.categorical_keys:
                    val = config.get(key)
                    max_val = self.encoders[key]['max']
                    vector.append(val / max_val if max_val > 0 else 0)

            # Encode Categorical (One-Hot)
            for key in self.categorical_keys:
                enc_info = self.encoders[key]
                val = str(config.get(key)) # Convert input to string to match key map
                
                # Create One-Hot Vector
                one_hot = [0] * enc_info['size']
                if val in enc_info['map']:
                    idx = enc_info['map'][val]
                    one_hot[idx] = 1
                vector.extend(one_hot)
            # getting the metric
            if with_metric: vector.append(config.get("metric")[0])
        
        else:
            for i in range(length):
                vector_i = []
                # Encode integer type, normalized as float
                for key in self.search_space:
                    if key not in self.categorical_keys:
                        val = config.get(key)[i]
                        max_val = self.encoders[key]['max']
                        vector_i.append(val / max_val if max_val > 0 else 0)

                # Encode Categorical (One-Hot)
                for key in self.categorical_keys:
                    enc_info = self.encoders[key]
                    val = str(config.get(key)[i]) # Convert input to string to match key map
                    
                    # Create One-Hot Vector
                    one_hot = [0] * enc_info['size']
                    if val in enc_info['map']:
                        idx = enc_info['map'][val]
                        one_hot[idx] = 1
                    vector_i.extend(one_hot)
                # getting the metric
                if with_metric: vector_i.append(config.get("metric")[0])
        
                vector.append(vector_i)
        return torch.tensor(vector, dtype=self.dtype)
    
    def __call__(self, config:Dict[str, Union[Any, Iterable]], with_metric:bool=False) -> torch.Tensor:
        """
        Encodes a model compression configuration
        """
        return self.encode(config, with_metric)
    