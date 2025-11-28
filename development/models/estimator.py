"""
@file estimator.py
@brief Neural Network Estimator for Hyperparameter Search.

This module defines a lightweight Multilayer Perceptron (MLP) used to approximate
the relationship between compression parameters (e.g., sparsity configurations) 
and model performance. It serves as a surrogate model to speed up the search 
process for optimal compression ratios.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .sequential import Sequential, Linear, ReLU

class Estimator:
    """
    A regression model used to predict the performance metric of a compressed model.
    
    This class wraps a custom MLP constructed using DMC's `Sequential` container.
    It includes built-in data normalization, training (fit), and inference (predict)
    routines tailored for the architecture search phase.
    """
    
    def __init__(
        self, 
        data: torch.Tensor, 
        hidden_dim: list[int] = [64, 128, 64], 
        dropout: float = 0.5, 
        device: str = "cpu"
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

        self.data = torch.tensor(data, dtype=torch.float32, device=self.device)

        self.x_mu = torch.tensor([0.0], device=device)
        self.x_std = torch.tensor([1.0], device=device)

        layers = []
        input_features = self.data.size(1) - 1

        # Input normalization (Affine=False acts as fixed scaling if params are frozen, 
        # but here it learns during training)
        layers.append(nn.BatchNorm1d(input_features, affine=False))
        layers.append(Linear(input_features, hidden_dim[0]))
        layers.append(nn.BatchNorm1d(hidden_dim[0]))
        layers.append(ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        input_features = hidden_dim[0]
        for h in hidden_dim[1:]:
            output_features = h

            layers.append(Linear(input_features, output_features))
            layers.append(nn.BatchNorm1d(output_features))
            layers.append(ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

            input_features = output_features
        layers.append(Linear(input_features, 1))
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
    def predict(self, X):
        """
        Predicts the metric for a given input configuration.
        """
        self.model.eval()
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(dtype=torch.float32, device=self.device)

        if X.dim() == 1:
            X = X.unsqueeze(0)
        return self.model(X).item()
