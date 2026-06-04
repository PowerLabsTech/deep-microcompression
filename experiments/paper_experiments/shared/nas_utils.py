"""
NAS estimator: definition, training, and the estimate() wrapper used by
evolutionary_search_compression_config().
"""

import os
from collections import defaultdict
from typing import Callable

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nas_data(nas_data_dir: str) -> dict:
    """
    Aggregate all .pth files in nas_data_dir into one combined dict.
    Each file must have been saved by generate_nas_data.py with the key
    'nas_parameters' mapping to a dict of {str: list}.
    """
    combined = defaultdict(list)
    files_loaded = 0

    for fname in sorted(os.listdir(nas_data_dir)):
        if not fname.endswith(".pth"):
            continue
        fpath = os.path.join(nas_data_dir, fname)
        data = torch.load(fpath, weights_only=False)
        nas_params = data["nas_parameters"]

        length = None
        for key, values in nas_params.items():
            combined[key].extend(values)
            if length is None:
                length = len(values)
            assert len(values) == length, \
                f"{fname}: inconsistent key lengths ({key}: {len(values)} vs {length})"

        files_loaded += 1

    if files_loaded == 0:
        raise FileNotFoundError(f"No .pth files found in {nas_data_dir}")

    n_samples = len(next(iter(combined.values())))
    print(f"Loaded {files_loaded} file(s) → {n_samples} total NAS samples.")
    return dict(combined)


# ---------------------------------------------------------------------------
# Estimator architecture
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


def build_estimator(input_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> nn.Module:
    """MLP with two residual blocks for predicting compressed-model accuracy."""

    class _Estimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            )
            self.res1 = _ResidualBlock(hidden_dim, dropout)
            self.res2 = _ResidualBlock(hidden_dim, dropout)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.res2(self.res1(self.stem(x))))

    return _Estimator()


# ---------------------------------------------------------------------------
# Estimator training
# ---------------------------------------------------------------------------

def train_estimator(
    encoded_data: torch.Tensor,
    device: str,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    epochs: int = 2000,
    batch_size: int = 128,
    val_split: float = 0.2,
    seed: int = 25,
):
    """
    Standardise encoded_data, do a train/val split, train the estimator with
    AdamW + CosineAnnealing, and restore the best checkpoint.

    Returns:
        model       – trained estimator (eval mode, on device)
        x_mu, x_std – feature standardisation parameters (on CPU)
        y_mu, y_std – target standardisation parameters (on CPU)
        history     – dict of loss/mae lists
    """
    torch.manual_seed(seed)

    X = encoded_data[:, :-1].float()
    Y = encoded_data[:, -1:].float()

    # Standardise
    x_mu  = X.mean(0);  x_std  = X.std(0).clamp(min=1e-10)
    y_mu  = Y.mean(0);  y_std  = Y.std(0).clamp(min=1e-10)
    X_norm = (X - x_mu) / x_std
    Y_norm = (Y - y_mu) / y_std

    # Split: first val_split fraction → val, rest → train
    n_val  = max(1, int(len(X_norm) * val_split))
    X_val,   Y_val   = X_norm[:n_val].to(device),  Y_norm[:n_val].to(device)
    X_train, Y_train = X_norm[n_val:].to(device),  Y_norm[n_val:].to(device)

    train_dl = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val,   Y_val),   batch_size=batch_size, shuffle=False)

    model     = build_estimator(X.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    l1_loss   = nn.L1Loss()

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val, best_state = float("inf"), None

    # Move standardisation params to device for MAE computation
    y_mu_d, y_std_d = y_mu.to(device), y_std.to(device)

    for epoch in range(epochs):
        model.train()
        t_loss = t_mae = 0.0
        for bx, by in train_dl:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                t_loss += loss.item()
                t_mae  += l1_loss(pred * y_std_d + y_mu_d,
                                  by   * y_std_d + y_mu_d).item()

        model.eval()
        v_loss = v_mae = 0.0
        with torch.no_grad():
            for bx, by in val_dl:
                pred   = model(bx)
                v_loss += criterion(pred, by).item()
                v_mae  += l1_loss(pred * y_std_d + y_mu_d,
                                  by   * y_std_d + y_mu_d).item()

        history["train_loss"].append(t_loss / len(train_dl))
        history["val_loss"].append(v_loss   / len(val_dl))
        history["train_mae"].append(t_mae   / len(train_dl))
        history["val_mae"].append(v_mae     / len(val_dl))

        scheduler.step()

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 200 == 0:
            print(f"  [{epoch+1:4d}/{epochs}] "
                  f"train_loss={history['train_loss'][-1]:.4f}  "
                  f"val_loss={history['val_loss'][-1]:.4f}  "
                  f"val_mae={history['val_mae'][-1]:.3f}%")

    model.load_state_dict(best_state)
    model.eval()
    return model, x_mu.cpu(), x_std.cpu(), y_mu.cpu(), y_std.cpu(), history


# ---------------------------------------------------------------------------
# estimate() wrapper for evolutionary search
# ---------------------------------------------------------------------------

def make_estimate_fn(
    estimator: nn.Module,
    nas_encoder,           # ConfigEncoder instance
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    y_mu: torch.Tensor,
    y_std: torch.Tensor,
    device: str,
) -> Callable:
    """
    Returns estimate(config) -> float, ready to pass to
    evolutionary_search_compression_config() as the second argument.
    """
    x_mu_d  = x_mu.to(device)
    x_std_d = x_std.to(device)
    y_mu_d  = y_mu.to(device)
    y_std_d = y_std.to(device)

    @torch.no_grad()
    def estimate(config: dict) -> float:
        enc    = nas_encoder(config, with_metric=False).float().to(device).unsqueeze(0)
        x_norm = (enc - x_mu_d) / x_std_d
        y_norm = estimator(x_norm)
        return (y_norm * y_std_d + y_mu_d).item()

    return estimate
