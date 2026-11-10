"""Normalização z-score por condição operacional.

O falso positivo instrumental de 03/06/2026 é, na origem, um artefato de
variância nula: a tag ``CFA`` esteve travada num único valor durante todo o
histórico de treino da condição ``P1_Q2``, de modo que o desvio-padrão daquela
condição foi exatamente zero e o limite de controle colapsou.

Um scaler global esconderia esse mecanismo — a tag tem variância global não-nula
porque muda de patamar entre condições. Normalizar por condição reproduz o
mecanismo no modelo global, o que permite medi-lo; e o piso de variância
(``sigma_floor_mode="global"``) é a correção proposta, avaliada por ablação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA_EPS = 1e-12


class OCScaler:
    """z-score por condição operacional, com piso opcional de desvio-padrão.

    Args:
        sigma_floor_mode:
            ``"none"``   — usa σ da própria condição (reproduz o mecanismo do
            falso positivo);
            ``"global"`` — usa ``max(σ_oc, σ_global)`` (correção recomendada);
            ``"eps"``    — usa ``max(σ_oc, eps · σ_global)``.
        eps: fração de σ_global usada no modo ``"eps"``.
        fallback_to_global: se uma condição não foi vista no ajuste, usa as
            estatísticas globais em vez de falhar.
    """

    def __init__(
        self,
        sigma_floor_mode: str = "none",
        eps: float = 0.1,
        fallback_to_global: bool = True,
    ) -> None:
        if sigma_floor_mode not in {"none", "global", "eps"}:
            raise ValueError(f"sigma_floor_mode inválido: {sigma_floor_mode}")
        self.sigma_floor_mode = sigma_floor_mode
        self.eps = eps
        self.fallback_to_global = fallback_to_global
        self.columns_: list[str] = []
        self.mu_: dict[str, np.ndarray] = {}
        self.sigma_: dict[str, np.ndarray] = {}
        self.sigma_raw_: dict[str, np.ndarray] = {}
        self.n_: dict[str, int] = {}
        self.mu_global_: np.ndarray | None = None
        self.sigma_global_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, oc: pd.Series) -> "OCScaler":
        """Ajusta média e desvio por condição operacional."""
        self.columns_ = list(df.columns)
        values = df.to_numpy(dtype=float)
        self.mu_global_ = np.nanmean(values, axis=0)
        self.sigma_global_ = np.nanstd(values, axis=0, ddof=1)

        oc_aligned = oc.reindex(df.index)
        for label, idx in df.groupby(oc_aligned.to_numpy()).groups.items():
            block = df.loc[idx].to_numpy(dtype=float)
            mu = np.nanmean(block, axis=0)
            sigma_raw = np.nanstd(block, axis=0, ddof=1) if len(block) > 1 else np.zeros(block.shape[1])
            self.mu_[label] = mu
            self.sigma_raw_[label] = sigma_raw
            self.sigma_[label] = self._apply_floor(sigma_raw)
            self.n_[label] = len(block)
        return self

    def _apply_floor(self, sigma_raw: np.ndarray) -> np.ndarray:
        """Aplica o piso de desvio-padrão conforme o modo configurado."""
        sigma = np.nan_to_num(sigma_raw, nan=0.0)
        if self.sigma_floor_mode == "global":
            sigma = np.maximum(sigma, self.sigma_global_)
        elif self.sigma_floor_mode == "eps":
            sigma = np.maximum(sigma, self.eps * self.sigma_global_)
        # Guarda numérica final: uma tag constante em todo o histórico (σ_global
        # também zero) não pode gerar divisão por zero.
        return np.where(sigma > SIGMA_EPS, sigma, 1.0)

    def transform(self, df: pd.DataFrame, oc: pd.Series) -> pd.DataFrame:
        """Aplica a normalização por condição, linha a linha."""
        if self.mu_global_ is None:
            raise RuntimeError("OCScaler.transform chamado antes de fit")

        values = df[self.columns_].to_numpy(dtype=float)
        oc_aligned = oc.reindex(df.index).to_numpy()
        out = np.empty_like(values)

        sigma_global_safe = np.where(self.sigma_global_ > SIGMA_EPS, self.sigma_global_, 1.0)
        for label in np.unique(oc_aligned):
            mask = oc_aligned == label
            if label in self.mu_:
                mu, sigma = self.mu_[label], self.sigma_[label]
            elif self.fallback_to_global:
                mu, sigma = self.mu_global_, sigma_global_safe
            else:
                raise KeyError(f"Condição operacional não vista no ajuste: {label}")
            out[mask] = (values[mask] - mu) / sigma

        return pd.DataFrame(out, index=df.index, columns=self.columns_)

    def fit_transform(self, df: pd.DataFrame, oc: pd.Series) -> pd.DataFrame:
        return self.fit(df, oc).transform(df, oc)

    def degenerate_report(self, n_unique: pd.Series | None = None) -> pd.DataFrame:
        """Pares (tag, condição) com desvio-padrão degenerado no ajuste.

        É a evidência quantitativa da recomendação de uma camada de validação de
        sensores a montante do modelo.
        """
        rows = []
        for label, sigma_raw in self.sigma_raw_.items():
            for j, col in enumerate(self.columns_):
                if sigma_raw[j] <= SIGMA_EPS:
                    rows.append(
                        {
                            "oc": label,
                            "tag": col,
                            "n": self.n_[label],
                            "sigma_oc": float(sigma_raw[j]),
                            "sigma_global": float(self.sigma_global_[j]),
                            "n_unique_global": int(n_unique[col]) if n_unique is not None else np.nan,
                        }
                    )
        return pd.DataFrame(rows).sort_values(["tag", "oc"]).reset_index(drop=True)
