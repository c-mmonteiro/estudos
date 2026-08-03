import torch
import torch.nn as nn
import time
import tqdm
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIGURAÇÃO
# =========================
class TrainerConfig:
    def __init__(
        self,
        epochs=200,
        lr=1e-3,
        optimizer="adam",
        loss="mse",
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss

    def to_dict(self):
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "device": str(self.device)
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            epochs=d["epochs"],
            lr=d["lr"],
            optimizer=d["optimizer"],
            loss=d["loss"],
            device=torch.device(d["device"])
        )

#########################################################
#####       Classe Residual (Skip Connection)      #####
#########################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout=0.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projeção da conexão residual, caso as dimensões sejam diferentes
        if in_dim == out_dim:
            self.shortcut = nn.Identity()

    def forward(self, x):
        if self.in_dim == self.out_dim:
            residual = self.shortcut(x)

        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)

        if self.in_dim == self.out_dim:
            return out + residual
        else:
            return out

# =========================
# MLP
# =========================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers=[45],
        activation="tanh",
        output_activation=None,
        dropout=0.0
    ):
        super().__init__()

        # salvar config do modelo
        self.config = {
            "input_dim": input_dim,
            "hidden_layers": hidden_layers,
            "activation": activation,
            "output_activation": output_activation,
            "dropout": dropout
        }

        layers = []
        prev_dim = input_dim
        act_fn = self._get_activation(activation)

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        if output_activation:
            layers.append(self._get_activation(output_activation))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def _get_activation(self, name):
        if name == "tanh":
            return nn.Tanh()
        elif name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {name}")


# =========================
# TRAINER
# =========================
class TorchTrainer:
    def __init__(self, model, config: TrainerConfig):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.loss_fn = self._build_loss()

        self.total_epochs = 0

    def _build_optimizer(self):
        if self.config.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError("Unknown optimizer")

    def _build_loss(self):
        if self.config.loss == "mse":
            return nn.MSELoss()
        elif self.config.loss == "mae":
            return nn.L1Loss()
        else:
            raise ValueError("Unknown loss")

    def fit(self, X, y, epochs=None):
        y = y.view(-1, 1)

        self.model.train()

        if epochs is None:
            n_epochs = self.config.epochs
        else:
            n_epochs = epochs

        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

        self.total_epochs += n_epochs

    def predict(self, X_tensor):
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor)

    def get_state(self):
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_epochs": self.total_epochs
        }

    def load_state(self, state):
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.total_epochs = state.get("total_epochs", 0)


# =========================
# ENSEMBLE - INFERÊNCIA
# =========================
class InferenceEnsemble:
    def __init__(self, n_models, model_fn, trainer_config):
        self.device = trainer_config.device
        self.trainers = []
        self.model_idx_data = []

        for _ in range(n_models):
            model = model_fn()
            trainer = TorchTrainer(model, trainer_config)
            self.trainers.append(trainer)

    def fit(self, X, y, ds_size=None, bootstrap=True, epochs=None, verbose=False):
        N = X.shape[0]

        if len(self.model_idx_data) == 0:
            if ds_size is None:
                ds_size = N

            for _ in range(len(self.trainers)):
                self.model_idx_data.append(torch.randint(0, N, (ds_size,), device=self.device))

        for idx, trainer in tqdm.tqdm(enumerate(self.trainers), disable=not verbose):
            if bootstrap:
                idx_data = self.model_idx_data[idx]
                trainer.fit(X[idx_data], y[idx_data], epochs=epochs)
            else:
                trainer.fit(X, y, epochs=epochs)

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        preds = torch.stack([
            trainer.predict(x).squeeze(-1)
            for trainer in self.trainers
        ])       

        if len(self.trainers) == 1:
            return preds.mean(dim=0), None

        else:
            return preds.mean(dim=0), preds.std(dim=0)
    
    def predict_quantiles(self, x, quantiles=[0.05, 0.95]):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        preds = torch.stack([
            trainer.predict(x).squeeze(-1)
            for trainer in self.trainers
        ])

        mean_pred = preds.mean(dim=0)
        q_values = torch.quantile(preds, quantiles, dim=0)

        return mean_pred, q_values
    
    def predict_sample(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        preds = torch.stack([
            trainer.predict(x).squeeze(-1)
            for trainer in self.trainers
        ])

        return preds

    def get_state(self):
        return {
            "trainers":[trainer.get_state() for trainer in self.trainers],
            "bootstrap_indices": self.model_idx_data
        }

    def load_state(self, states):
        for trainer, state in zip(self.trainers, states["trainers"]):
            trainer.load_state(state)
        self.model_idx_data = states.get("bootstrap_indices", [])


# =========================
# CLASSE DE ALTO NÍVEL
# =========================
class NNEnsambleModel:
    def __init__(
        self,
        model_fn,
        trainer_config,
        n_models=100,
        verbose=False
    ):
        self.device = trainer_config.device
        self.config = trainer_config

        self.inf_ens = InferenceEnsemble(n_models, model_fn, trainer_config)

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.scalers_fitted = False
        self.verbose = verbose

        self.n_models_ensemble = n_models

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def fit(self, X, y, ds_size=None, bootstrap=True, epochs=None):
        X_np = self._to_numpy(X)
        y_np = self._to_numpy(y)

        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)

        y_np = y_np.reshape(-1, 1)

        if not self.scalers_fitted:
            X_scaled = self.x_scaler.fit_transform(X_np)
            y_scaled = self.y_scaler.fit_transform(y_np).reshape(-1)
            self.scalers_fitted = True
        else:
            X_scaled = self.x_scaler.transform(X_np)
            y_scaled = self.y_scaler.transform(y_np).reshape(-1)

        X = self._to_tensor(X_scaled)
        y = self._to_tensor(y_scaled)

        start = time.time()

        if self.verbose:
            print("Treinando ensemble de inferência do valor...")

        self.inf_ens.fit(X, y, ds_size=ds_size, 
                         bootstrap=bootstrap, 
                         epochs=epochs, verbose=self.verbose)

        if self.verbose:
            print(f"Treinamento concluído em {time.time() - start:.2f}s")

    def predict(self, x, return_std=True, in_tensor=False):
        x_np = self._to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)

        x = self._to_tensor(self.x_scaler.transform(x_np))

        start = time.time()

        y_pred, std_pred = self.inf_ens.predict(x)


        y_pred_np = self._to_numpy(y_pred).reshape(-1, 1)
        y_pred_np = self.y_scaler.inverse_transform(y_pred_np).reshape(-1)

        if std_pred is not None:
            std_pred_np = self._to_numpy(std_pred).reshape(-1, 1)
            std_pred_np = std_pred_np * self.y_scaler.scale_[0]  # Ajusta a incerteza para a escala original do y
        else:
            std_pred_np = None
        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        if not return_std:
            if in_tensor:
                return self._to_tensor(y_pred_np)
            else:
                return self._to_numpy(y_pred)
        else:
            if in_tensor:
                return self._to_tensor(y_pred_np), self._to_tensor(std_pred_np)
            else:
                return y_pred_np, std_pred_np

    def predict_quantiles(self, x, alpha=0.05):
        x_np = self._to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
            
        x = self._to_tensor(self.x_scaler.transform(x_np))

        quantiles=[alpha/2, 1 - alpha/2]
        quantiles = self._to_tensor(quantiles)

        start = time.time()

        y_pred, q_values = self.inf_ens.predict_quantiles(x, quantiles)


        y_pred_np = self._to_numpy(y_pred).reshape(-1, 1)
        y_pred_np = self.y_scaler.inverse_transform(y_pred_np).reshape(-1)

        q_np = self._to_numpy(q_values)
        q_shape = q_np.shape
        q_values = self.y_scaler.inverse_transform(q_np).reshape(-1, 1)
        q_values = q_values.reshape(q_shape)


        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        return y_pred_np, q_values

    def predict_sample(self, x):
        x_np = self._to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        x = self._to_tensor(self.x_scaler.transform(x_np))

        start = time.time()

        y_pred = self.inf_ens.predict_sample(x)

        y_np = self._to_numpy(y_pred)
        y_shape = y_np.shape
        y_pred = self.y_scaler.inverse_transform(y_np.reshape(-1, 1)).reshape(y_shape)

        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        
        return y_pred

    # =========================
    # SAVE / LOAD
    # =========================
    def save(self, path):
        model_config = self.inf_ens.trainers[0].model.config

        state = {
            "model_config": model_config,
            "trainer_config": self.config.to_dict(),
            "uq_params": {
                "n_models": len(self.inf_ens.trainers)
            },
            "inf_ens": self.inf_ens.get_state(),
            "scalers": {
                "x_scaler": self.x_scaler,
                "y_scaler": self.y_scaler
            }
        }

        torch.save(state, path)

        if self.verbose:
            print(f"Modelo salvo em: {path}")

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        trainer_config = TrainerConfig.from_dict(checkpoint["trainer_config"])

        if device is not None:
            trainer_config.device = torch.device(device)

        model_config = checkpoint["model_config"]

        def model_fn():
            return MLP(**model_config)

        uq_params = checkpoint["uq_params"]

        uq_model = cls(
            model_fn=model_fn,
            trainer_config=trainer_config,
            n_models=uq_params["n_models"]
        )

        scalers = checkpoint.get("scalers")
        if scalers is not None:
            uq_model.x_scaler = scalers.get("x_scaler", uq_model.x_scaler)
            uq_model.y_scaler = scalers.get("y_scaler", uq_model.y_scaler)

        uq_model.inf_ens.load_state(checkpoint["inf_ens"])

        return uq_model