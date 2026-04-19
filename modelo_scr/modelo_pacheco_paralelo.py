import torch
import torch.nn as nn
import numpy as np


# =========================
# CONFIGURAÇÃO
# =========================
class TrainerConfigParallel:
    def __init__(
        self,
        epochs=200,
        lr=1e-3,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        self.epochs = epochs
        self.lr = lr


# =========================
# MLP PARALELO
# =========================
class ParallelMLP(nn.Module):
    def __init__(self, n_models, input_dim, hidden_layers=[45], activation="tanh"):
        super().__init__()

        self.n_models = n_models
        self.activation = self._get_activation(activation)

        dims = [input_dim] + hidden_layers + [1]

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.randn(n_models, dims[i], dims[i+1]) * 0.1)
            b = nn.Parameter(torch.zeros(n_models, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        # x: [n_models, batch, input_dim]

        for i in range(len(self.weights) - 1):
            x = torch.bmm(x, self.weights[i]) + self.biases[i].unsqueeze(1)
            x = self.activation(x)

        x = torch.bmm(x, self.weights[-1]) + self.biases[-1].unsqueeze(1)

        return x  # [n_models, batch, 1]

    def _get_activation(self, name):
        if name == "tanh":
            return torch.tanh
        elif name == "relu":
            return torch.relu
        elif name == "sigmoid":
            return torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {name}")


# =========================
# TRAINER PARALELO
# =========================
class ParallelTrainer:
    def __init__(self, model, config: TrainerConfigParallel):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def fit(self, X, y, input_std=None):
        device = self.config.device

        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        n_models = self.model.n_models
        n_samples, input_dim = X.shape

        for _ in range(self.config.epochs):
            self.optimizer.zero_grad()

            # =========================
            # RUÍDO POR MODELO
            # =========================
            if input_std is not None:
                noise = torch.randn(n_models, n_samples, input_dim, device=device) * input_std
                X_noisy = X.unsqueeze(0) + noise
            else:
                X_noisy = X.unsqueeze(0).expand(n_models, -1, -1)

            # =========================
            # BOOTSTRAP POR MODELO
            # =========================
            idx = torch.randint(0, n_samples, (n_models, n_samples), device=device)

            Xb = torch.gather(
                X_noisy,
                1,
                idx.unsqueeze(-1).expand(-1, -1, input_dim)
            )

            y_expand = y.view(1, -1, 1).expand(n_models, -1, -1)

            yb = torch.gather(
                y_expand,
                1,
                idx.unsqueeze(-1)
            )

            # =========================
            # FORWARD
            # =========================
            preds = self.model(Xb)

            # =========================
            # LOSS POR MODELO
            # =========================
            loss = (preds - yb) ** 2
            loss = loss.mean(dim=1)   # por modelo
            loss = loss.mean()        # escalar final

            loss.backward()
            self.optimizer.step()

    def predict_tensor(self, X):
        with torch.no_grad():
            preds = self.model(X)
        return preds


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
class InferenceEnsembleParallel:
    def __init__(self, n_models, model_fn, trainer_config):
        base_model = model_fn()

        hidden_layers = [
            layer.out_features
            for layer in base_model
            if isinstance(layer, nn.Linear)
        ][:-1]

        input_dim = base_model[0].in_features

        self.model = ParallelMLP(
            n_models=n_models,
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation="relu"
        )

        self.trainer = ParallelTrainer(self.model, trainer_config)
        self.device = trainer_config.device

    def fit(self, X, y):
        self.trainer.fit(X, y)

    def predict(self, x):
        x = torch.tensor([x], dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).expand(self.model.n_models, -1, -1)

        preds = self.trainer.predict_tensor(x)
        return preds.mean().item()


# =========================
# ENSEMBLE - INCERTEZA
# =========================
class UncertaintyEnsembleParallel:
    def __init__(self, n_models, model_fn, trainer_config):
        base_model = model_fn()

        hidden_layers = [
            layer.out_features
            for layer in base_model
            if isinstance(layer, nn.Linear)
        ][:-1]

        input_dim = base_model[0].in_features

        self.model = ParallelMLP(
            n_models=n_models,
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            activation="relu"
        )

        self.trainer = ParallelTrainer(self.model, trainer_config)
        self.device = trainer_config.device

    def fit(self, X, y, input_std):
        self.trainer.fit(X, y, input_std=input_std)

    def predict_uncertainty(self, x, mcs_samples, input_std, u_M, k=2):
        x_mc = monte_carlo_sample(x, input_std, mcs_samples, self.device)

        # expandir para todos modelos
        x_mc = x_mc.unsqueeze(0).expand(self.model.n_models, -1, -1)

        preds = self.trainer.predict_tensor(x_mc)  # [n_models, mcs, 1]
        preds = preds.reshape(-1)

        u_E = torch.std(preds, unbiased=True)
        u_cI = torch.sqrt(torch.tensor(u_M, device=self.device)**2 + u_E**2)

        return (k * u_cI).item()


# =========================
# EXEMPLO DE USO
# =========================
import time
if __name__ == "__main__":

    X = np.random.rand(200, 3)
    y = np.random.rand(200)
    
    start_time = time.time()

    def model_fn():
        return nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    config = TrainerConfigParallel(epochs=100, lr=1e-3)

    # Inferência
    inf = InferenceEnsembleParallel(30, model_fn, config)
    inf.fit(X, y)

    # Incerteza
    unc = UncertaintyEnsembleParallel(100, model_fn, config)
    unc.fit(X, y, input_std=0.01)

    x_new = np.array([0.5, 0.2, 0.1])

    y_pred = inf.predict(x_new)

    U_I = unc.predict_uncertainty(x_new, 1000, 0.01, 0.038)

    end_time = time.time()
    print(f"Tempo total: {end_time - start_time:.2f} segundos")

    print("Pred:", y_pred)
    print("Unc:", U_I)