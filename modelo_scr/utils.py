import torch
import numpy as np
import plotly.graph_objects as go
from modelo_scr.modelo_pacheco import *


class Dataset:
    def __init__(self, modelo_math, val_max, val_min, num_amostras, 
                 erro_sistematico_x, erro_aleatorio_x, dist_erro_x, 
                 erro_sistematico_y, erro_aleatorio_y, dist_erro_y):
        self.val_max = val_max
        self.val_min = val_min
        self.num_amostras = num_amostras
        self.erro_sistematico_x = erro_sistematico_x
        self.erro_aleatorio_x = erro_aleatorio_x
        self.dist_erro_x = dist_erro_x
        self.erro_sistematico_y = erro_sistematico_y
        self.erro_aleatorio_y = erro_aleatorio_y
        self.dist_erro_y = dist_erro_y
        self.modelo_math = modelo_math

        #Valores verdadeiros de X e Y
        self.X_true = torch.linspace(val_max, val_min, num_amostras)
        self.y_true = self.modelo_math(self.X_true)

        #Valores medidos de X
        if dist_erro_x == "uniforme":
            rand_error = self.X_true*self.erro_aleatorio_x*(torch.rand_like(self.X_true)*2 - 1)
        elif dist_erro_x == "normal":
            rand_error = self.X_true*self.erro_aleatorio_x*torch.randn_like(self.X_true)
        else:
            raise ValueError("dist_erro_x deve ser 'uniforme' ou 'normal'")
        
        self.X_measured = self.X_true + rand_error + self.X_true*self.erro_sistematico_x

        print(f'Shape X_measured: {tuple(self.X_measured.shape)}')

        #########################################################################
        #Valores Calculados de Y (Valor verdadeiro para entrada X_measured)
        self.y_calc = self.modelo_math(self.X_measured)

        #Valores medidos de Y
        if dist_erro_y == "uniforme":
            rand_error = self.y_true*self.erro_aleatorio_y*(torch.rand_like(self.y_true)*2 - 1)
        elif dist_erro_y == "normal":
            rand_error = self.y_true*self.erro_aleatorio_y*torch.randn_like(self.y_true)
        else:
            raise ValueError("dist_erro_y deve ser 'uniforme' ou 'normal'")
        
        self.y_measured = self.y_true + rand_error + self.y_true*self.erro_sistematico_y

        print(f'Shape y_measured: {tuple(self.y_measured.shape)}')

        

    def gerar_monte_carlo(self, num_mc_simulation_samples):
        # Monte Carlo de X_measured
        self.X_measured_mc = self.X_measured.repeat_interleave(num_mc_simulation_samples, dim=0)
        
        #Gera o ruído para X_measured_mc de acordo com a distribuição especificada
        if self.dist_erro_x == "uniforme":
            X_noise = self.X_measured_mc*self.erro_aleatorio_x*(torch.rand_like(self.X_measured_mc)*2 - 1)
        elif self.dist_erro_x == "normal":
            X_noise = self.X_measured_mc*self.erro_aleatorio_x*torch.randn_like(self.X_measured_mc)
        else:
            raise ValueError("dist_erro_x deve ser 'uniforme' ou 'normal'")
        
        self.X_measured_mc = self.X_measured_mc + X_noise

        print(f'Shape X_measured_mc: {tuple(self.X_measured_mc.shape)}')

        ###################################################################################
        #Monte Carlo de y_measured
        self.y_measured_mc = self.y_measured.repeat_interleave(num_mc_simulation_samples, dim=0)

        #Gera o ruído para y_measured_mc de acordo com a distribuição especificada
        if self.dist_erro_y == "uniforme":
            y_noise = self.y_measured_mc*self.erro_aleatorio_y*(torch.rand_like(self.y_measured_mc)*2 - 1)
        elif self.dist_erro_y == "normal":
            y_noise = self.y_measured_mc*self.erro_aleatorio_y*torch.randn_like(self.y_measured_mc)
        else:
            raise ValueError("dist_erro_y deve ser 'uniforme' ou 'normal'")

        self.y_measured_mc = self.y_measured_mc + y_noise

        print(f'Shape y_measured_mc: {tuple(self.y_measured_mc.shape)}')


    def plot_dataset(self, fig=None):
        if fig is None:
            fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.X_true.numpy(),
                                 y=self.y_calc.numpy(),
                mode='markers', marker=dict(color='yellow', size=7, opacity=0.8),
                name='Calculated'))

        fig.add_trace(go.Scatter(x=self.X_true.numpy(),
                                 y=self.y_measured.numpy(),
                mode='markers', marker=dict(color='blue', size=3, opacity=0.7),
                name='Measured'))

        fig.add_trace(go.Scatter(x=self.X_true.numpy(), 
                                 y=self.y_true.numpy(),
                mode='lines', marker=dict(color='red', size=1, opacity=0.1),
                name='True'))

        fig.add_trace(go.Scatter(x=self.X_measured.numpy(), 
                                 y=self.y_measured.numpy(),
                mode='markers', marker=dict(color='grey', size=5, opacity=0.8),
                name='All Measured'))

        fig.add_vline(x=self.val_min,
                line_dash="dash", line_color="green",
                annotation_text=f"Faixa dos dados", annotation_position="top right")
        fig.add_vline(x=self.val_max,
                line_dash="dash", line_color="green")

        fig.update_layout(title='True vs Measured',
        xaxis_title='X', yaxis_title='Y',
        template='plotly_white'
        )

        fig.show()
    
    def plot_monte_carlo(self, fig=None):

        if fig is None:
            fig = go.Figure()

        # Valor Monte Carlo
        fig.add_trace(go.Scatter(
            x=self.X_measured_mc,
            y=self.y_measured_mc,
            mode='markers',
            name='Monte Carlo',
            marker=dict(color='grey', size=5, opacity=0.7)
        ))

        # Valor de calibração original
        fig.add_trace(go.Scatter(
            x=self.X_measured,
            y=self.y_measured,
            mode='markers',
            name='Medição Original',
            marker=dict(color='blue', size=5, opacity=0.7)
        ))

        # Valor Verdadeiro
        fig.add_trace(go.Scatter(
            x=self.X_measured,
            y=self.y_true,
            mode='markers',
            name='Valor Verdadeiro',
            marker=dict(color='red', size=5, opacity=0.7)
        ))

        fig.update_layout(
            title=f'Conjunto com Simulação de Monte Carlo',
            xaxis_title='X',
            yaxis_title='Y'
        )
        fig.show()

##################################################################
#############       UncertaintyEvaluator        ##################
##################################################################           

class UncertaintyEvaluator:
    def __init__(self, model, cp_model_dict, test_dataset, alpha=0.05, num_mc_simulations=1000, verbose=True):
        self.model = model
        self.cp_model_dict = cp_model_dict
        self.test_dataset = test_dataset
        self.alpha = alpha
        self.num_mc_simulations = num_mc_simulations
        self.verbose = verbose

        self.y_pred = {}
        self.lower_quantiles = {}
        self.upper_quantiles = {}

        
        #Model Baseline
        y_pred_model, quantiles = self.model.predict_quantiles(self.test_dataset.X_measured.reshape(-1, 1).numpy(), 
                                                   alpha=self.alpha)
        self.y_pred["model"] = y_pred_model
        self.lower_quantiles["model"] = quantiles[0]
        self.upper_quantiles["model"] = quantiles[1]

        #Conformal Prediction Models
        for name, cp_model in self.cp_model_dict.items():
            y_pred_cp, lower_cp, upper_cp = cp_model.predict(self.test_dataset.X_measured.reshape(-1, 1).numpy())
            self.y_pred[name] = y_pred_cp
            self.lower_quantiles[name] = lower_cp
            self.upper_quantiles[name] = upper_cp


        #Monte Carlo
        y_pred_mc, lower_mc, upper_mc = self.intervalo_mc_quantis_torch(self.test_dataset.X_measured.reshape(-1, 1).numpy(),
                                                                        self.model,
                                                                        n_simulacoes=self.num_mc_simulations,
                                                                        quantis=[self.alpha/2, 1 - self.alpha/2],
                                                                        erro_aleatorio=self.test_dataset.erro_aleatorio_x,
                                                                        erro_sistematico=self.test_dataset.erro_sistematico_x,
                                                                        distribuicao=self.test_dataset.dist_erro_x)
        self.y_pred["monte_carlo"] = y_pred_mc
        self.lower_quantiles["monte_carlo"] = lower_mc
        self.upper_quantiles["monte_carlo"] = upper_mc

        #Avaliação das métricas de incerteza
        self.coverage = {}
        self.avg_width = {}
        if self.verbose:
            print(f'Model       |   Coverage   |   Average Width')
            print(f'-'*50)
        
        for name, y_pred in self.y_pred.items():
            self.coverage[name], self.avg_width[name] = self.uncertainty_metrics(self.lower_quantiles[name], self.upper_quantiles[name], self.test_dataset.y_true)
            if self.verbose:
                print(f"{name.ljust(12)}|{100*self.coverage[name]:13.4f}%|{self.avg_width[name]:14.4f}")

    #################################################################
    ##########      Monte Carlo                           ###########
    #################################################################

    def intervalo_mc_quantis_torch(self,
                                    X,
                                    model,
                                    n_simulacoes=1000,
                                    quantis=[0.025, 0.975],
                                    erro_aleatorio=0.15,
                                    erro_sistematico=0.0,
                                    distribuicao="uniforme"):

        # Converte X para NumPy sem usar operacoes torch
        if hasattr(X, "detach"):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        X_np = np.asarray(X_np, dtype=np.float32).reshape(-1, 1)
        n_amostras = X_np.shape[0]

        # Repete cada amostra n_simulacoes vezes
        X_mc = np.repeat(X_np, repeats=n_simulacoes, axis=0)

        # Ruido multiplicativo
        if distribuicao == "uniforme":
            eps = np.random.uniform(-1.0, 1.0, size=X_mc.shape).astype(np.float32)
        elif distribuicao == "gaussiano":
            eps = np.random.normal(0.0, 1.0, size=X_mc.shape).astype(np.float32)
        else:
            raise ValueError("distribuicao deve ser 'uniforme' ou 'gaussiano'.")

        X_mc = X_mc + X_mc * erro_aleatorio * eps #- X_mc*erro_sistematico

        # Inferencia
        y_pred = model.predict_sample(X_mc)


        y_mc = np.asarray(y_pred, dtype=np.float32).reshape(model.n_models_ensemble, n_amostras, n_simulacoes)
        y_mc = y_mc.transpose(1, 2, 0) # Transposição para (n_amostras, n_simulacoes, n_models_ensemble)
        y_mc = y_mc.reshape(n_amostras, n_simulacoes * model.n_models_ensemble) # Reshape para (n_amostras, n_simulacoes * n_models_ensemble)
          
        y_low = np.quantile(y_mc, quantis[0], axis=1)
        y_high = np.quantile(y_mc, quantis[1], axis=1)
        y_mean = y_mc.mean(axis=1)

        return y_mean, y_low, y_high

    #################################################################
    ##########      Gráficos de incerteza e métricas      ###########
    #################################################################        

    def uncertainty_metrics(self, lower, upper, y_calc):
        # Cobertura efetiva: fração de y_calc dentro da banda
        y_calc_np = y_calc.numpy().flatten()
        coverage = np.mean((y_calc_np >= lower) & (y_calc_np <= upper))

        # Largura média da banda
        avg_width = np.mean(upper - lower)

        return coverage, avg_width
    
    def graph_with_uncertainty(self, model_name, fig=None):
        
        # Ordenar por X para o gráfico ficar correto
        sort_idx = np.argsort(self.test_dataset.X_measured.numpy().flatten())
        x_sorted = self.test_dataset.X_measured.numpy().flatten()[sort_idx]
        val_sorted = self.y_pred[model_name][sort_idx]
        lower_sorted = self.lower_quantiles[model_name][sort_idx]
        upper_sorted = self.upper_quantiles[model_name][sort_idx]
        y_true_sorted = self.test_dataset.y_true.numpy().flatten()[sort_idx]

        if fig is None:
            fig = go.Figure()

        # Banda de incerteza
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_sorted, x_sorted[::-1]]),
            y=np.concatenate([lower_sorted, upper_sorted[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Incerteza (quantis)'
        ))

        # Predição
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=val_sorted,
            mode='lines',
            name='Predição',
            line=dict(color='blue')
        ))

        # Valor calculado de teste
        fig.add_trace(go.Scatter(
            x=x_sorted,
            y=y_true_sorted,
            mode='markers',
            name='Referência (Calculado)',
            marker=dict(color='red', size=5, opacity=0.7)
        ))

        fig.add_vline(x=self.test_dataset.val_min,
                    line_dash="dash", line_color="green",
                    annotation_text=f"Faixa de calibração", annotation_position="top right")
        fig.add_vline(x=self.test_dataset.val_max,
                    line_dash="dash", line_color="green")

        fig.update_layout(
            title=f'Predição com Banda de Incerteza  |  Cobertura efetiva: {self.coverage[model_name]:.1%}  |  Largura média: {self.avg_width[model_name]:.4f}',
            xaxis_title='X',
            yaxis_title='Y'
        )
        fig.show()