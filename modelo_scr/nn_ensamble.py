import torch
import torch.nn as nn
import time
import tqdm

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

    def fit(self, X, y):
        y = y.view(-1, 1)

        self.model.train()

        for _ in range(self.config.epochs):
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

    def predict(self, X_tensor):
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor)

    def get_state(self):
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }

    def load_state(self, state):
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])


# =========================
# ENSEMBLE - INFERÊNCIA
# =========================
class InferenceEnsemble:
    def __init__(self, n_models, model_fn, trainer_config):
        self.device = trainer_config.device
        self.trainers = []

        for _ in range(n_models):
            model = model_fn()
            trainer = TorchTrainer(model, trainer_config)
            self.trainers.append(trainer)

    def fit(self, X, y, ds_size=None):
        N = X.shape[0]

        if ds_size is None:
            ds_size = N

        for trainer in tqdm.tqdm(self.trainers):
            idx = torch.randint(0, N, (ds_size,), device=self.device)
            trainer.fit(X[idx], y[idx])

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        preds = torch.stack([
            trainer.predict(x).squeeze(-1)
            for trainer in self.trainers
        ])

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

    def get_state(self):
        return [trainer.get_state() for trainer in self.trainers]

    def load_state(self, states):
        for trainer, state in zip(self.trainers, states):
            trainer.load_state(state)


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

        self.verbose = verbose

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def fit(self, X, y, ds_size=None):
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        start = time.time()

        if self.verbose:
            print("Treinando ensemble de inferência do valor...")

        self.inf_ens.fit(X, y, ds_size=ds_size)

        if self.verbose:
            print(f"Treinamento concluído em {time.time() - start:.2f}s")

    def predict(self, x, return_std=True, in_tensor=False):
        x = self._to_tensor(x)

        start = time.time()

        y_pred, std_pred = self.inf_ens.predict(x)

        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        if not return_std:
            if in_tensor:
                return y_pred
            else:
                return self._to_numpy(y_pred)
        else:
            if in_tensor:
                return y_pred, std_pred
            else:
                return self._to_numpy(y_pred), self._to_numpy(std_pred)

    def predict_quantiles(self, x, alpha=0.05):
        x = self._to_tensor(x)
        quantiles=[alpha/2, 1 - alpha/2]
        quantiles = self._to_tensor(quantiles)

        start = time.time()

        y_pred, q_values = self.inf_ens.predict_quantiles(x, quantiles)

        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        return self._to_numpy(y_pred), self._to_numpy(q_values)

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
            "inf_ens": self.inf_ens.get_state()
        }

        torch.save(state, path)

        if self.verbose:
            print(f"Modelo salvo em: {path}")

    @classmethod
    def load(cls, path, device=None):
        checkpoint = torch.load(path, map_location=device)

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

        uq_model.inf_ens.load_state(checkpoint["inf_ens"])

        return uq_model