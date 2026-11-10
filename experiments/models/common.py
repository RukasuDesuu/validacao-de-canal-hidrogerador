"""Infraestrutura compartilhada pelos modelos: protocolo, limiares e janelas.

Duas escolhas valem registro porque diferem do pipeline de produção:

1. Os limites de controle são calculados **exclusivamente** sobre o bloco de
   treino. Em produção o ``health_data`` é calculado e descartado (o ``fit``
   recebe a série completa), de modo que os limites incorporam os próprios
   eventos que deveriam detectar.
2. O limiar é por condição operacional. O erro de reconstrução é heterocedástico
   entre regimes; um limiar único faria a condição de maior variância dominar a
   taxa de alarme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.config import CALIBRATION_FRAC, NUM_STDS, PROTOCOLS, STRIDE, TRAIN_STRIDE, WINDOW
from experiments.data import Dataset
from experiments.oc_scaler import OCScaler


@dataclass
class WindowSet:
    """Janelas de um protocolo, já normalizadas por condição operacional."""

    X: np.ndarray  # (n_windows, window, n_features)
    index: pd.DatetimeIndex  # timestamp do último ponto de cada janela
    oc: np.ndarray  # condição operacional da janela
    train_mask: np.ndarray  # bloco de saúde inteiro
    fit_mask: np.ndarray  # ajuste do modelo (início do bloco, subamostrado)
    cal_mask: np.ndarray  # calibração do limiar (trecho final, fora da amostra)
    score_mask: np.ndarray
    scaler: OCScaler

    @property
    def X_train(self) -> np.ndarray:
        """Janelas de ajuste do modelo (subamostradas)."""
        return self.X[self.fit_mask]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    def flat(self) -> np.ndarray:
        return self.X.reshape(len(self.X), -1)


def protocol_bounds(protocol: str) -> tuple[tuple[str, str], tuple[str, str] | None]:
    p = PROTOCOLS[protocol]
    return p["train"], p["score"]


def build_windows(
    ds: Dataset,
    protocol: str,
    sigma_floor_mode: str = "none",
    window: int = WINDOW,
    stride: int = STRIDE,
) -> WindowSet:
    """Normaliza por condição operacional (ajuste só no treino) e janela a série.

    O rótulo de condição de uma janela é o da sua última amostra, coerente com o
    ``MWAutoEncoder``, que indexa cada janela pelo ponto final.
    """
    train_bounds, score_bounds = protocol_bounds(protocol)
    idx = ds.index
    train_rows = (idx >= pd.Timestamp(train_bounds[0])) & (idx <= pd.Timestamp(train_bounds[1]))

    scaler = OCScaler(sigma_floor_mode=sigma_floor_mode)
    scaler.fit(ds.df.loc[train_rows], ds.oc)
    scaled = scaler.transform(ds.df, ds.oc)

    from experiments.data import make_windows

    X, last_idx = make_windows(scaled.to_numpy(), ds.segments, window, stride)
    last_ts = idx[last_idx]
    oc_win = ds.oc.to_numpy()[last_idx]

    train_mask = np.asarray((last_ts >= pd.Timestamp(train_bounds[0])) & (last_ts <= pd.Timestamp(train_bounds[1])))
    if score_bounds is None:
        score_mask = np.ones(len(last_ts), dtype=bool)
    else:
        score_mask = np.asarray(
            (last_ts >= pd.Timestamp(score_bounds[0])) & (last_ts <= pd.Timestamp(score_bounds[1]))
        )

    # Divisão temporal do bloco de saúde: o modelo ajusta no início e o limiar é
    # calibrado no trecho final, que o modelo nunca viu. As janelas de ajuste são
    # ainda subamostradas, porque com passo de 1 h janelas vizinhas compartilham
    # 95,8% das amostras.
    train_idx = np.flatnonzero(train_mask)
    cut = int(len(train_idx) * (1 - CALIBRATION_FRAC))
    # A janela de ajuste mais tardia não pode se sobrepor à primeira janela de
    # calibração, sob pena de vazamento de amostras.
    fit_idx = train_idx[:cut][:: TRAIN_STRIDE]
    cal_idx = train_idx[cut + window :]

    fit_mask = np.zeros_like(train_mask)
    fit_mask[fit_idx] = True
    cal_mask = np.zeros_like(train_mask)
    cal_mask[cal_idx] = True

    return WindowSet(
        X=X,
        index=last_ts,
        oc=oc_win,
        train_mask=train_mask,
        fit_mask=fit_mask,
        cal_mask=cal_mask,
        score_mask=score_mask,
        scaler=scaler,
    )


def ucl_per_oc(
    scores_train: np.ndarray,
    oc_train: np.ndarray,
    num_stds: float = NUM_STDS,
    min_points: int = 30,
) -> tuple[dict[str, float], float]:
    """Limite de controle ``média + k·σ`` por condição operacional.

    Returns:
        ``(mapa_oc_para_ucl, ucl_global)``. O global cobre condições com poucos
        pontos de treino.
    """
    finite = np.isfinite(scores_train)
    global_ucl = float(np.mean(scores_train[finite]) + num_stds * np.std(scores_train[finite], ddof=1))

    table: dict[str, float] = {}
    for label in np.unique(oc_train):
        sel = (oc_train == label) & finite
        if sel.sum() < min_points:
            continue
        vals = scores_train[sel]
        table[str(label)] = float(vals.mean() + num_stds * vals.std(ddof=1))
    return table, global_ucl


def apply_ucl(oc: np.ndarray, table: dict[str, float], global_ucl: float) -> np.ndarray:
    """Vetor de limiares alinhado às janelas."""
    return np.array([table.get(str(label), global_ucl) for label in oc], dtype=float)


def finalize(
    ws: WindowSet,
    scores: np.ndarray,
    num_stds: float = NUM_STDS,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Aplica o protocolo: limiares do treino, recorte da janela de pontuação."""
    table, global_ucl = ucl_per_oc(scores[ws.cal_mask], ws.oc[ws.cal_mask], num_stds=num_stds)
    ucl = apply_ucl(ws.oc, table, global_ucl)

    keep = ws.score_mask
    extra = {
        "n_train_windows": int(ws.train_mask.sum()),
        "n_fit_windows": int(ws.fit_mask.sum()),
        "n_cal_windows": int(ws.cal_mask.sum()),
        "n_scored_windows": int(keep.sum()),
        "n_oc_with_ucl": len(table),
        "ucl_global": global_ucl,
        "num_stds": num_stds,
    }
    return ws.index[keep], scores[keep], ucl[keep], ws.oc[keep], extra
