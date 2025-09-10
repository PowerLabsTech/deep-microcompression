import torch
from torch import nn

from .sequential import Sequential, Linear, ReLU

class Estimator:

    def __init__(self, data, hidden_dim = [64, 128, 64], dropout=.5, device="cpu") -> None:
        self.device = device

        self.data = torch.tensor(data, dtype=torch.float32, device=self.device)

        self.x_mu = torch.tensor([0.0], device=device)
        self.x_std = torch.tensor([1.0], device=device)

        layers = []
        input_features = self.data.size(1) - 1
        for h in hidden_dim:
            output_features = h
            layers.append(Linear(input_features, output_features))
            layers.append(ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

            input_features = output_features
        layers.append(Linear(input_features, 1))
        self.model = Sequential(*layers).to(self.device)

        self.optimizer_fun = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion_fun = nn.MSELoss()


    def _normalize(self, X):
        # move to correct device
        X = X.to(self.device)
        return (X - self.x_mu) / self.x_std

    def fit(self, epochs, batch_size=32):
        X, Y = self.data[:, :-1], self.data[:, -1].unsqueeze(dim=1)
        
        self.x_mu = X.mean(dim=0).to(self.device)
        x_std = X.std(dim=0)
        self.x_std = torch.where(x_std > 1e-10, x_std, torch.ones_like(x_std)).to(self.device)

        X = self._normalize(X)

        N = X.size(0)
        assert N > 1, "The number of data point is less than 2, it can not be split."
        idx = torch.randperm(N, device=self.device)

        n_val = max(1, int(N * 0.2))  # at least 1 val sample if N > 1
        n_val = min(n_val, N - 1)     # but leave at least 1 for training

        val_idx, train_idx = idx[:n_val], idx[n_val:]
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        
        history = self.model.fit(
            train_dataloader=(X_train, Y_train), 
            epochs=epochs, 
            criterion_fun=self.criterion_fun, 
            optimizer_fun=self.optimizer_fun, 
            validation_dataloader=(X_val, Y_val),
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
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = self._normalize(X)
        return self.model(X).item()


    # def get_nas_prune_channel(
    #     self,
    #     input_shape, 
    #     data_loader, 
    #     metric_fun, 
    #     device="cpu",
    #     num_data=100
    # ) -> "Sequential":
    #     prune_channel_hp = self.get_prune_channel_possible_hypermeters()
    #     param = []

    #     for _ in range(num_data):
    #         prune_param_config = dict()
    #         prune_param = list()
    #         for layer_name, layer_prune_channel_hp in prune_channel_hp.items():
    #             random_layer_prune_channel_hp = random.choice(layer_prune_channel_hp)
    #             prune_param.append(random_layer_prune_channel_hp)
    #             prune_param_config[layer_name] = random_layer_prune_channel_hp
    #         print(prune_param_config)
    #         compression_config = {
    #                 "prune_channel" :{
    #                     "sparsity" : prune_param_config,
    #                     "metric" : "l2"
    #                 },
    #             }
            
    #         prune_channel_model = self.init_compress(config=compression_config, input_shape=input_shape)
    #         prune_channel_model_metric = prune_channel_model.evaluate(data_loader=data_loader, metrics={"metric": metric_fun}, device=device)

    #         param.append(prune_param + [prune_channel_model_metric["metric"]])

    #     data = torch.Tensor(param)

    #     X, Y = data[:,:-1], data[:,-1]
        
    #     x_mu = X.mean(dim=0)
    #     x_std = X.std(dim=0)
    #     x_std = torch.where(x_std >1e-10, x_std, torch.ones_like(x_std))
    #     X = (X - x_mu) / x_std

    #     model = Sequential(
    #         Linear(X.size(1), 100),
    #         ReLU(),
    #         Linear(100, 100),
    #         ReLU(),
    #         Linear(100, 1)
    #     )
    #     model.to(device)

    #     optimizer_fun = torch.optim.Adam(model.parameters(), lr=1e-3)
    #     criterion_fun = nn.MSELoss()

    #     model.fit(
    #         train_dataloader=(X, Y), epochs=100, 
    #         criterion_fun=criterion_fun, optimizer_fun=optimizer_fun, 
    #         device=device, batch_size=64
    #     )


    #     return model
