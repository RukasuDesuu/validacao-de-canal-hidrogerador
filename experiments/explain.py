"""Explicabilidade por SHAP, agregada sobre as janelas de um evento.

A versão anterior explicava uma única janela — a de erro máximo — e descrevia o
resultado como consistente. Uma amostra de tamanho 1 não sustenta essa palavra.
Aqui as atribuições são calculadas para todas as janelas do evento e reportadas
como mediana com intervalo interquartil, o que também expõe a dispersão
introduzida pelo ``GradientExplainer``, que é estocástico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.config import MASK_RATIO
from experiments.data import Dataset
from experiments.ground_truth import EVENTS_BY_ID
from experiments.models.mae_global import DEVICE, torch_model_for_shap

BACKGROUND_N = 50
MAX_EVENT_WINDOWS = 48
# Amostras do estimador por janela. O padrão da biblioteca (200) multiplicado
# pelo número de janelas torna o cálculo proibitivo em CPU; 64 mantém a mediana
# estável e a dispersão entre janelas continua sendo reportada.
NSAMPLES = 64


class ReconstructionErrorScorer(nn.Module):
    """Expõe o erro de reconstrução como saída escalar diferenciável."""

    def __init__(self, mae: nn.Module) -> None:
        super().__init__()
        self.mae = mae
        self.mae.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.mae(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2)).unsqueeze(1)


def shap_for_event(
    ds: Dataset,
    event_id: str,
    protocol: str = "P1",
    mask_ratio: float = MASK_RATIO,
    seed: int = 42,
    model_ws=None,
) -> pd.DataFrame:
    """Atribuições por sensor nas janelas de um evento.

    Returns:
        DataFrame indexado por tag, com mediana, quartis e a soma de |SHAP| sobre
        os passos de tempo de cada janela.
    """
    import shap

    model, ws = model_ws if model_ws else torch_model_for_shap(ds, protocol, mask_ratio=mask_ratio, seed=seed)
    ev = EVENTS_BY_ID[event_id]

    sel = (ws.index >= pd.Timestamp(ev.start)) & (ws.index <= pd.Timestamp(ev.end))
    idx = np.flatnonzero(np.asarray(sel))
    if len(idx) == 0:
        raise ValueError(f"nenhuma janela do evento {event_id} disponível")
    if len(idx) > MAX_EVENT_WINDOWS:
        idx = idx[np.linspace(0, len(idx) - 1, MAX_EVENT_WINDOWS).astype(int)]

    scorer = ReconstructionErrorScorer(model).to(DEVICE)
    background = torch.tensor(ws.X[ws.fit_mask][:BACKGROUND_N], dtype=torch.float32).to(DEVICE)
    explainer = shap.GradientExplainer(scorer, background)

    windows = torch.tensor(ws.X[idx], dtype=torch.float32).to(DEVICE)
    values = explainer.shap_values(windows, nsamples=NSAMPLES)
    arr = np.asarray(values[0] if isinstance(values, list) else values)
    # (n_janelas, tempo, sensores[, saída]) -> soma de |.| sobre o tempo
    if arr.ndim == 4:
        arr = arr[..., 0]
    per_window = np.abs(arr).sum(axis=1)

    tags = list(ws.scaler.columns_)
    df = pd.DataFrame(per_window, columns=tags)
    out = pd.DataFrame(
        {
            "median": df.median(),
            "q1": df.quantile(0.25),
            "q3": df.quantile(0.75),
            "mean": df.mean(),
        }
    )
    out["n_windows"] = len(idx)
    return out.sort_values("median", ascending=False)


def shap_by_cluster(per_tag: pd.DataFrame, clusters: list[dict]) -> pd.DataFrame:
    """Agrega as atribuições por cluster de sintoma."""
    rows = []
    for cluster in clusters:
        tags = [t for t in cluster["tags"] if t in per_tag.index]
        if not tags:
            continue
        rows.append(
            {
                "abbrev": cluster["abbrev"],
                "name": cluster["name"],
                "median": float(per_tag.loc[tags, "median"].sum()),
                "n_tags": len(tags),
            }
        )
    return pd.DataFrame(rows).sort_values("median", ascending=False).reset_index(drop=True)
