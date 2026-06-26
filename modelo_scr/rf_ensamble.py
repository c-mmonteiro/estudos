from sklearn.ensemble import RandomForestRegressor 
import numpy as np

class RFEnsambleModel:
    def __init__(self, X_train, y_train, n_models_ensemble=100, num_amostras_treino=None):
        self.model = RandomForestRegressor(n_estimators=n_models_ensemble, 
                                           random_state=42, 
                                           max_samples=num_amostras_treino, 
                                           verbose=True)

        if isinstance(X_train, np.ndarray):
            X_train_np = X_train
        else:
            X_train_np = X_train.numpy()

        if isinstance(y_train, np.ndarray):
            y_train_np = y_train
        else:
            y_train_np = y_train.numpy()

        self.model.fit(X_train_np, y_train_np)
        self.n_models_ensemble = n_models_ensemble

    def predict(self, X):
        tree_probs = np.vstack([tree.predict(X) for tree in self.model.estimators_])

        mean_prob = tree_probs.mean(axis=0)
        std_prob = tree_probs.std(axis=0)

        return mean_prob, std_prob

    def predict_quantiles(self, X, alpha=0.05):
        quantiles=[alpha/2, 1 - alpha/2]
        # Para RandomForestRegressor, podemos usar o método `predict` para obter a média e o método `predict` com `return_std=True` para obter a incerteza
        tree_probs = np.vstack([tree.predict(X) for tree in self.model.estimators_])
        mean_prob = tree_probs.mean(axis=0)
        quantiles = np.quantile(tree_probs, quantiles, axis=0)

        return mean_prob, quantiles

    def predict_sample(self, X):
        # Para RandomForestRegressor, podemos usar o método `predict` para obter a média e o método `predict` com `return_std=True` para obter a incerteza
        tree_probs = np.vstack([tree.predict(X) for tree in self.model.estimators_])

        return tree_probs