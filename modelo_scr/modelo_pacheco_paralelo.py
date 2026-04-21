import torch
import torch.nn as nn
import time


# =========================
# CONFIG
# =========================
class TrainerConfig:
    def __init__(self, epochs=200, lr=1e-3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")
        self.epochs = epochs
        self.lr = lr


# =========================
# CAMADA PARALELA
# =========================
class ParallelLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(n_models, in_features, out_features) * 0.01
        )
        self.bias = nn.Parameter(
            torch.zeros(n_models, 1, out_features)
        )

    def forward(self, x):
        # x: [M, B, in]
        return torch.bmm(x, self.weight) + self.bias


# =========================
# MLP PARALELO
# =========================
class ParallelMLP(nn.Module):
    def __init__(self, n_models, input_dim, hidden_layers=[64, 32], activation="relu"):
        super().__init__()

        self.n_models = n_models
        self.layers = nn.ModuleList()

        prev = input_dim

        for h in hidden_layers:
            self.layers.append(ParallelLinear(n_models, prev, h))
            prev = h

        self.out = ParallelLinear(n_models, prev, 1)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError("Activation not supported")

    def forward(self, x):
        # x: [B, d] → [M, B, d]
        x = x.unsqueeze(0).repeat(self.n_models, 1, 1)

        for layer in self.layers:
            x = self.act(layer(x))

        x = self.out(x)

        return x  # [M, B, 1]


# =========================
# TRAINER PARALELO
# =========================
class ParallelTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.device = config.device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.epochs = config.epochs

    def fit(self, X, y):
        B = X.shape[0]

        # expand y → [M, B, 1]
        y = y.view(1, B, 1).repeat(self.model.n_models, 1, 1)

        for _ in range(self.epochs):
            self.optimizer.zero_grad()

            preds = self.model(X)

            loss = self.loss_fn(preds, y)

            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self.model(X)  # [M, B, 1]


# =========================
# MONTE CARLO
# =========================
def monte_carlo_sample_batch(X, std, n_samples, device):
    B, d = X.shape

    X_expanded = X.unsqueeze(1).repeat(1, n_samples, 1)
    noise = torch.randn((B, n_samples, d), device=device) * std

    X_mc = X_expanded + noise

    return X_mc.view(-1, d), B


# =========================
# MODELO COMPLETO
# =========================
class UQModel_Paralelo:
    def __init__(
        self,
        input_dim,
        hidden_layers=[64, 32],
        n_models=100,
        trainer_config=None,
        mcs_samples=1000,
        input_std=0.01,
        u_M=0.0,
        k=2,
        verbose=False
    ):
        self.device = trainer_config.device

        self.model = ParallelMLP(
            n_models=n_models,
            input_dim=input_dim,
            hidden_layers=hidden_layers
        )

        self.trainer = ParallelTrainer(self.model, trainer_config)

        self.mcs_samples = mcs_samples
        self.input_std = input_std
        self.u_M = u_M
        self.k = k
        self.verbose = verbose

    def _to_tensor(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _to_numpy(self, x):
        return x.detach().cpu().numpy()

    def fit(self, X, y):
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        start = time.time()

        self.trainer.fit(X, y)

        if self.verbose:
            print(f"Treinamento concluído em {time.time() - start:.2f}s")

    def predict(self, X, return_uncertainty=True):
        X = self._to_tensor(X)

        if X.dim() == 1:
            X = X.unsqueeze(0)

        start = time.time()

        preds = self.trainer.predict(X)  # [M, B, 1]

        mean = preds.mean(dim=0).squeeze()

        if not return_uncertainty:
            return self._to_numpy(mean)

        # MCS
        X_mc, B = monte_carlo_sample_batch(
            X,
            self.input_std,
            self.mcs_samples,
            self.device
        )

        preds_mc = self.trainer.predict(X_mc)  # [M, B*mcs, 1]

        preds_mc = preds_mc.view(
            self.model.n_models,
            B,
            self.mcs_samples
        )

        preds_mc = preds_mc.reshape(-1, B)

        u_E = torch.std(preds_mc, dim=0)

        u_M_t = torch.tensor(self.u_M, device=self.device)

        u_cI = torch.sqrt(u_M_t**2 + u_E**2)

        U = self.k * u_cI

        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        return self._to_numpy(mean), self._to_numpy(U)