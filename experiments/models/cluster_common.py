"""Base comum aos baselines por cluster x condição operacional.

Reproduz a estrutura do pipeline de produção — um modelo por par (cluster,
condição) — com uma diferença deliberada: os limites de controle saem apenas do
bloco de saúde do protocolo, e não da série completa.

Limitação herdada, registrada porque afeta a leitura dos resultados: a janela
móvel dentro de uma condição operacional **não é** um intervalo contíguo no
tempo, e sim uma vizinhança de mesmo regime, possivelmente separada por semanas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.config import PROTOCOLS
from experiments.data import Dataset


def cluster_frames(
    ds: Dataset,
    cluster: dict,
    protocol: str,
    window: int,
    min_extra: int = 30,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Pares (condição, quadro de saúde, quadro de pontuação) de um cluster.

    Condições sem pontos suficientes para formar janelas de saúde são puladas —
    é o mesmo critério de ``min_training_points`` do orquestrador.
    """
    cfg = PROTOCOLS[protocol]
    tags = [t for t in cluster["tags"] if t in ds.df.columns]
    if not tags:
        return []

    idx = ds.index
    train_rows = np.asarray((idx >= pd.Timestamp(cfg["train"][0])) & (idx <= pd.Timestamp(cfg["train"][1])))
    if cfg["score"] is None:
        score_rows = np.ones(len(idx), dtype=bool)
    else:
        score_rows = np.asarray((idx >= pd.Timestamp(cfg["score"][0])) & (idx <= pd.Timestamp(cfg["score"][1])))

    out = []
    oc = ds.oc.to_numpy()
    for label in np.unique(oc):
        in_oc = oc == label
        health = ds.df.loc[in_oc & train_rows, tags]
        score = ds.df.loc[in_oc & score_rows, tags]
        if len(health) < window + min_extra or len(score) <= window:
            continue
        out.append((str(label), health, score))
    return out


def aggregate_cluster_ratios(per_cluster: dict[str, pd.Series], index: pd.DatetimeIndex) -> np.ndarray:
    """Score do sistema = maior razão ``métrica/limiar`` entre os clusters.

    Um único escalar por hora, comparável aos modelos globais depois de
    normalizado pelo limiar (que passa a ser 1,0 por construção).
    """
    if not per_cluster:
        return np.full(len(index), np.nan)
    frame = pd.DataFrame(per_cluster).reindex(index)
    return frame.max(axis=1).to_numpy()


def first_valid_index(series: np.ndarray, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Índice restrito às horas com pelo menos um cluster pontuado."""
    return index[np.isfinite(series)]
