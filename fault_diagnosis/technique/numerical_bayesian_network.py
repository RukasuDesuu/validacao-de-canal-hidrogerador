"""Implementação numérica vetorizada de Rede Bayesiana.

Modificado para incluir FAR/FDR e métodos de detecção.
"""

import numpy as np
import pandas as pd


class NumericalBayesianNetwork:
    """Implementação numérica vetorizada de Rede Bayesiana para diagnóstico de falhas.

    Otimizações:
    - Operações matriciais NumPy (sem loops Python)
    - Log-probabilidades para estabilidade numérica
    - Pré-computação de matrizes de log

    Attributes:
        failure_modes: Array com nomes dos modos de falha
        symptoms: Lista com nomes dos sintomas
        cpt: Matriz CPT shape (n_failure_modes, n_symptoms)
        log_P: Matriz log(CPT + epsilon)
        log_1_minus_P: Matriz log(1 - CPT + epsilon)
        fdr_threshold: Threshold para detecção de falha (padrão 0.9)
        far_threshold: Threshold complementar para falso alarme (1 - fdr_threshold)

    """

    def __init__(
        self,
        data: pd.DataFrame,
        fdr_threshold: float = 0.9,
        epsilon: float = 1e-10,
        mode: str = "marginal",
        background_prob: float | np.ndarray = 0.05,
    ):
        """Inicializa a rede Bayesiana com pré-computação de log-matrizes.

        Args:
            data: DataFrame com coluna "Modo de Falha" + colunas de sintomas
            fdr_threshold: Threshold para considerar detecção de falha (padrão 0.9)
            epsilon: Valor pequeno para evitar log(0)
            mode: Modo de inferência - "marginal" (independente por FM, soma pode
                  exceder 1.0) ou "categorical" (normalizada, soma=1)
            background_prob: Probabilidade de fundo para o modo marginal. Escalar ou
                  array com shape (n_symptoms,). Usado na hipótese ¬FM_i.

        """
        # Extrair dados
        self.failure_modes = data["Modo de Falha"].values
        self.symptoms = data.columns[1:].tolist()

        # Matriz CPT: shape (n_failure_modes, n_symptoms)
        self.cpt = data.drop("Modo de Falha", axis=1).values.astype(np.float64)

        # Pré-computar log-matrizes (otimização principal)
        # Isso é feito uma única vez na inicialização
        self.log_P = np.log(self.cpt + epsilon)
        self.log_1_minus_P = np.log(1 - self.cpt + epsilon)

        # Metadados
        self.n_failure_modes = len(self.failure_modes)
        self.n_symptoms = len(self.symptoms)
        self.epsilon = epsilon

        # Modo de inferência e probabilidade de fundo
        self.mode = mode
        bg = np.full(self.n_symptoms, background_prob) if np.isscalar(background_prob) else np.asarray(background_prob)
        self.background_prob = bg
        self.log_bg = np.log(bg + epsilon)
        self.log_1_minus_bg = np.log(1 - bg + epsilon)

        # Thresholds FAR/FDR (compatibilidade com ProbabilisticExpertSystem)
        self.fdr_threshold = fdr_threshold
        self.far_threshold = 1 - fdr_threshold
        self.initial_probability = 1 / self.n_failure_modes

        # Cache para máscara de sintomas (otimização adicional)
        self._symptom_to_idx = {s: i for i, s in enumerate(self.symptoms)}

    def _symptoms_to_mask(self, active_symptoms: list[str]) -> np.ndarray:
        """Convert list of symptoms to boolean mask.

        Args:
            active_symptoms: Lista de nomes de sintomas ativos

        Returns:
            Array booleano shape (n_symptoms,) onde True indica sintoma ativo

        """
        mask = np.zeros(self.n_symptoms, dtype=bool)
        for symptom in active_symptoms:
            if symptom in self._symptom_to_idx:
                mask[self._symptom_to_idx[symptom]] = True
        return mask

    def calculate_probabilities(self, active_symptoms: list[str], mode: str | None = None) -> np.ndarray:
        """Calcula P(FM | symptoms) usando operações vetorizadas em log-space.

        Args:
            active_symptoms: Lista de sintomas ativos
            mode: Modo de inferência ("categorical" ou "marginal"). Se None, usa self.mode.

        Returns:
            Array de probabilidades para cada modo de falha

        """
        mode = mode or self.mode
        mask = self._symptoms_to_mask(active_symptoms)
        if mode == "marginal":
            return self._calculate_marginal(mask)
        return self._calculate_categorical(mask)

    def _calculate_log_likelihood(self, mask: np.ndarray) -> np.ndarray:
        """Calcula log-verossimilhança por FM dado máscara de sintomas.

        Args:
            mask: Array booleano shape (n_symptoms,) onde True indica sintoma ativo

        Returns:
            Array shape (n_failure_modes,) com log-verossimilhança por FM

        """
        if mask.any():
            log_active = self.log_P[:, mask].sum(axis=1)
        else:
            log_active = 0

        if (~mask).any():
            log_inactive = self.log_1_minus_P[:, ~mask].sum(axis=1)
        else:
            log_inactive = 0

        return log_active + log_inactive

    def _calculate_categorical(self, mask: np.ndarray) -> np.ndarray:
        """Calcula probabilidades categóricas (normalizada, soma=1).

        Implementa softmax sobre log-verossimilhanças:
        P(FM_i | S) = exp(log_like_i) / Σ_j exp(log_like_j)

        Args:
            mask: Array booleano shape (n_symptoms,)

        Returns:
            Array de probabilidades normalizadas (soma = 1.0)

        """
        log_numerators = self._calculate_log_likelihood(mask)

        # Log-sum-exp trick para normalização numericamente estável
        max_log = log_numerators.max()
        log_denominator = max_log + np.log(np.exp(log_numerators - max_log).sum())

        return np.exp(log_numerators - log_denominator)

    def _calculate_marginal(self, mask: np.ndarray) -> np.ndarray:
        """Calcula probabilidades marginais (independente por FM, soma pode exceder 1.0).

        Para cada FM, aplica Bayes binário: FM_i vs ¬FM_i (hipótese de fundo).
        P(FM_i | S) = σ(log_like_i - log_like_neg) = 1 / (1 + exp(log_like_neg - log_like_i))

        Args:
            mask: Array booleano shape (n_symptoms,)

        Returns:
            Array de probabilidades marginais por FM

        """
        # Verossimilhança por FM (mesmo cálculo da categórica)
        log_like = self._calculate_log_likelihood(mask)

        # Verossimilhança da hipótese de fundo (¬FM)
        log_like_neg = 0
        if mask.any():
            log_like_neg = log_like_neg + self.log_bg[mask].sum()
        if (~mask).any():
            log_like_neg = log_like_neg + self.log_1_minus_bg[~mask].sum()

        # Sigmoid: P(FM_i|S) = 1 / (1 + exp(-(log_like - log_like_neg)))
        log_odds = log_like - log_like_neg
        return 1.0 / (1.0 + np.exp(-log_odds))

    def get_probabilities_dataframe(self, active_symptoms: list[str], mode: str | None = None) -> pd.DataFrame:
        """Retorna probabilidades em formato DataFrame (compatível com ProbabilisticExpertSystem).

        Args:
            active_symptoms: Lista de sintomas ativos
            mode: Modo de inferência ("categorical" ou "marginal"). Se None, usa self.mode.

        Returns:
            DataFrame com colunas ['Modo de Falha', 'Probability'], ordenado por probabilidade

        """
        probs = self.calculate_probabilities(active_symptoms, mode=mode)

        df = pd.DataFrame({"Modo de Falha": self.failure_modes, "Probability": probs})

        df = df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
        df["Probability"] = df["Probability"].round(4)

        return df

    def get_detection_result_single(
        self, active_symptoms: list[str], mode: str | None = None
    ) -> dict[str, float | str]:
        """Decide se há detecção de falha ou falso alarme para um conjunto de sintomas.

        Aplica os thresholds FDR e FAR para determinar se a probabilidade é suficiente
        para considerar uma detecção válida.

        Args:
            active_symptoms: Lista de sintomas ativos
            mode: Modo de inferência ("categorical" ou "marginal"). Se None, usa self.mode.

        Returns:
            dict: Dicionário com:
                - "Detected Fault": modo de falha detectado ou "False Alarm"
                - "Probability": probabilidade associada
                - "FDR": taxa de detecção (para detecções válidas)
                - "FAR": taxa de falso alarme

        """
        probs_df = self.get_probabilities_dataframe(active_symptoms, mode=mode)
        fault = probs_df.iloc[0]["Modo de Falha"]
        prob = probs_df.iloc[0]["Probability"]

        if prob >= self.fdr_threshold:
            return {
                "Detected Fault": fault,
                "Probability": prob,
                "FDR": prob,
                "FAR": self.far_threshold,
            }
        else:
            return {
                "Detected Fault": "False Alarm",
                "Probability": prob,
                "FDR": 0,
                "FAR": 1,
            }

    def process_predictions(
        self, cluster_predictions: dict[str, list[int]], mode: str | None = None
    ) -> list[dict[str, float]]:
        """Processa predições para identificar sintomas ativos e calcular probabilidades de falha.

        Implementação vetorizada: constrói matriz de observações (n_timesteps, n_symptoms)
        e executa inferência em batch com operações matriciais NumPy, evitando loop Python
        por timestep.

        Args:
            cluster_predictions: Dicionário com equipamentos como chaves e listas de sintomas ativos como valores
                Exemplo: {"Vibração": [0, 1, 1], "Temperatura": [1, 0, 1]}
            mode: Modo de inferência ("categorical" ou "marginal"). Se None, usa self.mode.

        Returns:
            list: Lista de dicts {failure_mode: probability} para cada timestep

        """
        mode = mode or self.mode

        n_timesteps = len(cluster_predictions[next(iter(cluster_predictions))])

        # Matriz de observações: shape (n_timesteps, n_symptoms)
        obs = np.zeros((n_timesteps, self.n_symptoms), dtype=np.float64)
        for symptom_name, timeseries in cluster_predictions.items():
            if symptom_name in self._symptom_to_idx:
                obs[:, self._symptom_to_idx[symptom_name]] = np.asarray(timeseries, dtype=np.float64)

        # Log-verossimilhança: (n_timesteps, n_failure_modes)
        # log_P shape: (n_failure_modes, n_symptoms) → obs @ log_P.T
        log_like = obs @ self.log_P.T + (1.0 - obs) @ self.log_1_minus_P.T

        if mode == "marginal":
            # Hipótese de fundo: escalar por timestep (n_timesteps,)
            log_like_neg = obs @ self.log_bg + (1.0 - obs) @ self.log_1_minus_bg
            log_odds = log_like - log_like_neg[:, np.newaxis]
            probs = 1.0 / (1.0 + np.exp(-log_odds))  # sigmoid vetorizado
        else:
            # Categorical: softmax por timestep
            max_log = log_like.max(axis=1, keepdims=True)
            exp_log = np.exp(log_like - max_log)
            probs = exp_log / exp_log.sum(axis=1, keepdims=True)

        probs = np.round(probs, 4)

        # Converter para lista de dicts (formato de saída original)
        return [dict(zip(self.failure_modes, probs[i])) for i in range(n_timesteps)]

    def get_detection_result(
        self, cluster_predictions: dict[str, list[int]], mode: str | None = None
    ) -> list[dict[str, float | str]]:
        """Decide se há detecção de falha ou falso alarme aplicando FDR/FAR para múltiplos timesteps.

        Args:
            cluster_predictions: Dicionário com sintomas e séries temporais de anomalia
            mode: Modo de inferência ("categorical" ou "marginal"). Se None, usa self.mode.

        Returns:
            list: Lista de dicionários com resultado de detecção para cada timestep

        """
        raw_predictions = self.process_predictions(cluster_predictions, mode=mode)
        detection_results = []

        for raw in raw_predictions:
            fault, prob = max(raw.items(), key=lambda x: x[1])
            if prob >= self.fdr_threshold:
                detection_results.append(
                    {
                        "Detected Fault": fault,
                        "Probability": prob,
                        "FDR": prob,
                        "FAR": self.far_threshold,
                    }
                )
            else:
                detection_results.append({"Detected Fault": "False Alarm", "Probability": prob, "FDR": 0, "FAR": 1})

        return detection_results

    def evaluate_detection_rates(
        self, actual_faults: list[str], predicted_results: list[dict[str, float | str]]
    ) -> dict[str, float]:
        """Compara detecções preditas com ground truth e calcula FDR/FAR.

        Calcula métricas de desempenho do sistema de detecção:
        - FDR (Fault Detection Rate): TP / (TP + FN)
        - FAR (False Alarm Rate): FP / (FP + TN)

        Args:
            actual_faults: Lista de modos de falha reais para cada instância
                          Use "No Fault" para indicar ausência de falha
            predicted_results: Resultados de detecção obtidos via get_detection_result()

        Returns:
            dict: Dicionário com FDR e FAR calculados

        """
        TP = FP = TN = FN = 0

        for actual, pred in zip(actual_faults, predicted_results):
            predicted_fault = pred["Detected Fault"]

            if actual != "No Fault":
                # Há uma falha real
                if predicted_fault == actual:
                    TP += 1  # Detecção correta
                else:
                    FN += 1  # Falha não detectada
            else:
                # Não há falha real
                if predicted_fault == "False Alarm":
                    TN += 1  # Corretamente identificou ausência de falha
                else:
                    FP += 1  # Falso alarme

        fdr_value = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        far_value = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        return {"FDR": round(fdr_value, 4), "FAR": round(far_value, 4)}

    def __repr__(self) -> str:
        """Return string representation of NumericalBayesianNetwork."""
        return (
            f"NumericalBayesianNetwork("
            f"failure_modes={self.n_failure_modes}, "
            f"symptoms={self.n_symptoms}, "
            f"fdr_threshold={self.fdr_threshold}, "
            f"mode={self.mode!r}, "
            f"epsilon={self.epsilon})"
        )
