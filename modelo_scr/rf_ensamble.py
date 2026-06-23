from sklearn.ensemble import RandomForestRegressor 
import numpy as np

class RFEnsambleModel:
    def __init__(self, X_train, y_train, n_models_ensemble=100, num_amostras_treino=None):
        self.model = RandomForestRegressor(n_estimators=n_models_ensemble, 
                                           random_state=42, 
                                           max_samples=num_amostras_treino, 
                                           verbose=True)
        
        self.model.fit(X_train.reshape(-1, 1).numpy(), 
                       y_train.numpy())

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