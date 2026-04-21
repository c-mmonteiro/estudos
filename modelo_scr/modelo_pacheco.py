import torch
import torch.nn as nn
import time


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


# =========================
# MONTE CARLO
# =========================
def monte_carlo_sample(x, std, n_samples, device):
    x = x.unsqueeze(0).repeat(n_samples, 1)
    noise = torch.randn_like(x, device=device) * std
    return x + noise


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

    def fit(self, X, y):
        N = X.shape[0]

        for trainer in self.trainers:
            idx = torch.randint(0, N, (N,), device=self.device)
            trainer.fit(X[idx], y[idx])

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        preds = torch.stack([trainer.predict(x).squeeze(-1) for trainer in self.trainers])

        return preds.mean(dim=0)


# =========================
# ENSEMBLE - INCERTEZA
# =========================
class UncertaintyEnsemble:
    def __init__(self, n_models, model_fn, trainer_config):
        self.device = trainer_config.device
        self.config = trainer_config
        self.trainers = []

        for _ in range(n_models):
            model = model_fn()
            trainer = TorchTrainer(model, trainer_config)
            self.trainers.append(trainer)

    def fit(self, X, y, input_std, mcs_samples=50):
        N = X.shape[0]

        for trainer in self.trainers:

            X_mcs_list = []
            y_mcs_list = []

            for i in range(N):
                x_mc = monte_carlo_sample(
                    X[i],
                    input_std,
                    mcs_samples,
                    self.device
                )

                X_mcs_list.append(x_mc)

                y_i = torch.full((mcs_samples,), y[i], device=self.device)
                y_mcs_list.append(y_i)

            X_mcs = torch.cat(X_mcs_list, dim=0)
            y_mcs = torch.cat(y_mcs_list, dim=0)

            idx = torch.randint(0, X_mcs.shape[0], (N,), device=self.device)

            trainer.fit(X_mcs[idx], y_mcs[idx])

    def predict_uncertainty(self, x, mcs_samples, input_std, u_M, k=2):
        x_mc = monte_carlo_sample(x, input_std, mcs_samples, self.device)

        outputs = torch.zeros(0, device=self.device)

        for trainer in self.trainers:
            preds = trainer.predict(x_mc).view(-1)
            outputs = torch.cat((outputs, preds))

        u_E = torch.std(outputs, unbiased=True)
        u_M_t = torch.tensor(u_M, device=self.device)

        u_cI = torch.sqrt(u_M_t**2 + u_E**2)

        return (k * u_cI).item()


# =========================
# CLASSE DE ALTO NÍVEL
# =========================
class UQModel:
    def __init__(
        self,
        model_fn,
        trainer_config,
        n_models=100,
        mcs_samples=1000,
        input_std=0.01,
        u_M=0.0,
        k=2,
        verbose=False
    ):
        self.device = trainer_config.device
        self.config = trainer_config

        self.inf_ens = InferenceEnsemble(n_models, model_fn, trainer_config)
        self.unc_ens = UncertaintyEnsemble(n_models, model_fn, trainer_config)

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
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def fit(self, X, y):
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        start = time.time()

        if self.verbose:
            print("Treinando ensemble de inferência do valor...")

        self.inf_ens.fit(X, y)

        if self.verbose:
            print("Treinando ensemble de incerteza da incerteza...")

        self.unc_ens.fit(X, y, input_std=self.input_std)

        if self.verbose:
            print(f"Calibração concluída em {time.time() - start:.2f}s")

    def predict(self, x, return_uncertainty=True):
        x = self._to_tensor(x)

        start = time.time()

        y_pred = self.inf_ens.predict(x)

        if not return_uncertainty:
            if self.verbose:
                print(f"Inferência em {time.time() - start:.4f}s")
            return self._to_numpy(y_pred)

        # Compute uncertainty per sample
        U_list = []
        for i in range(x.shape[0]):
            U_i = self.unc_ens.predict_uncertainty(
                x[i],
                mcs_samples=self.mcs_samples,
                input_std=self.input_std,
                u_M=self.u_M,
                k=self.k
            )
            U_list.append(U_i)

        U_I = torch.tensor(U_list, device=self.device)

        if self.verbose:
            print(f"Inferência em {time.time() - start:.4f}s")

        return self._to_numpy(y_pred), self._to_numpy(U_I)