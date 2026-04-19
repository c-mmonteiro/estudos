import torch
import torch.nn as nn
import numpy as np


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


# =========================
# MLP FLEXÍVEL
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
# TRAINER (fit incremental)
# =========================
class TorchTrainer:
    def __init__(self, model, config: TrainerConfig):
        self.model = model.to(config.device)
        self.config = config

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
        X = torch.tensor(X, dtype=torch.float32, device=self.config.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.config.device).view(-1, 1)

        self.model.train()

        for _ in range(self.config.epochs):
            self.optimizer.zero_grad()
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            self.optimizer.step()

    # 🔥 interno (sem conversão)
    def predict_tensor(self, X_tensor):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)
        return preds

    # API externa
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.config.device)
        preds = self.predict_tensor(X)
        return preds.cpu().numpy()


# =========================
# BOOTSTRAP
# =========================
def bootstrap_sample(X, y):
    idx = np.random.choice(len(X), size=len(X), replace=True)
    return X[idx], y[idx]


# =========================
# MONTE CARLO
# =========================
def monte_carlo_sample(x, std, n_samples, device):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.unsqueeze(0).repeat(n_samples, 1)

    noise = torch.randn_like(x) * std

    return x + noise  # [n_samples, input_dim]


# =========================
# ENSEMBLE - INFERÊNCIA
# =========================
class InferenceEnsemble:
    def __init__(self, n_models, model_fn, trainer_config):
        self.models = []
        self.trainers = []
        self.n_models = n_models

        for _ in range(n_models):
            model = model_fn()
            trainer = TorchTrainer(model, trainer_config)

            self.models.append(model)
            self.trainers.append(trainer)

    def fit(self, X, y):
        for trainer in self.trainers:
            Xb, yb = bootstrap_sample(X, y)
            trainer.fit(Xb, yb)

    def predict(self, x):
        preds = []

        for trainer in self.trainers:
            x_tensor = torch.tensor([x], dtype=torch.float32, device=trainer.config.device)
            pred = trainer.predict_tensor(x_tensor)
            preds.append(pred.item())

        return float(np.mean(preds))


# =========================
# ENSEMBLE - INCERTEZA
# =========================
class UncertaintyEnsemble:
    def __init__(self, n_models, model_fn, trainer_config):
        self.models = []
        self.trainers = []
        self.config = trainer_config

        for _ in range(n_models):
            model = model_fn()
            trainer = TorchTrainer(model, trainer_config)

            self.models.append(model)
            self.trainers.append(trainer)

    def fit(self, X, y, input_std):
        for trainer in self.trainers:
            X_noisy = X + np.random.normal(0, input_std, X.shape)
            Xb, yb = bootstrap_sample(X_noisy, y)
            trainer.fit(Xb, yb)

    def predict_uncertainty(self, x, mcs_samples, input_std, u_M, k=2):
        device = self.config.device

        # Monte Carlo batch (GPU)
        x_mc = monte_carlo_sample(x, input_std, mcs_samples, device)

        outputs = []

        for trainer in self.trainers:
            preds = trainer.predict_tensor(x_mc)
            outputs.append(preds.view(-1))

        outputs = torch.cat(outputs)

        # estatística em torch
        u_E = torch.std(outputs, unbiased=True)
        u_cI = torch.sqrt(torch.tensor(u_M, device=device)**2 + u_E**2)
        U_I = k * u_cI

        return U_I.item()


# =========================
# EXEMPLO DE USO
# =========================
import time
if __name__ == "__main__":

    # Dados fictícios
    X = np.random.rand(200, 3)
    y = np.random.rand(200)

    start_time = time.time()

    # Definição do modelo
    def model_fn():
        return MLP(
            input_dim=3,
            hidden_layers=[64, 32],
            activation="relu",
            dropout=0
        )

    # Configuração
    config = TrainerConfig(
        epochs=100,
        lr=1e-3
    )

    # =========================
    # INFERÊNCIA
    # =========================
    inf_ens = InferenceEnsemble(
        n_models=30,
        model_fn=model_fn,
        trainer_config=config
    )
    inf_ens.fit(X, y)

    # =========================
    # INCERTEZA
    # =========================
    unc_ens = UncertaintyEnsemble(
        n_models=100,
        model_fn=model_fn,
        trainer_config=config
    )
    unc_ens.fit(X, y, input_std=0.01)

    # Novo ponto
    x_new = np.array([0.5, 0.2, 0.1])

    # Predição
    y_pred = inf_ens.predict(x_new)

    # Incerteza
    U_I = unc_ens.predict_uncertainty(
        x_new,
        mcs_samples=1000,
        input_std=0.01,
        u_M=0.038
    )

    end_time = time.time()
    print(f"Tempo total: {end_time - start_time:.2f} segundos")

    print("Predição:", y_pred)
    print("Incerteza:", U_I)