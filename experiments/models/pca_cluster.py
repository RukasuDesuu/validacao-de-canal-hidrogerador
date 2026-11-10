"""RMWPCA_SC por cluster x condição operacional — o baseline real de produção.

Duas leituras são gravadas:

* ``rmwpca_q``   — razão ``Q/UCL_Q``, análoga ao erro de reconstrução dos demais
  modelos e diretamente comparável a eles;
* ``rmwpca_pcw`` — número de componentes de PCW fora da banda, com limiar em 1,5,
  reproduzindo a regra "2 de 3" que a produção de fato usa para decidir.

A distinção importa: o argumento de que este baseline é cego a um degrau
permanente é sobre o autocentramento por janela, que afeta as duas leituras, mas
só a segunda corresponde ao alarme que a operação vê.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from experiments.cache import save_scores
from experiments.config import NUM_STDS, PCA_CLUSTER_WINDOW
from experiments.data import Dataset
from experiments.models.cluster_common import aggregate_cluster_ratios, cluster_frames


def _fit_one(health: pd.DataFrame, score: pd.DataFrame, window: int):
    """Ajusta um RMWPCA_SC numa condição e pontua a série daquela condição."""
    from fault_diagnosis.technique.rmwpcasc import RMWPCA_SC

    model = RMWPCA_SC()
    ref = health.iloc[:window].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(health, ref_data=ref, window_size=window)
        res = model.transform_current_data(score)

    limits = model.limits or {}
    q_high = float(limits.get("Q", {}).get("high", np.nan))
    pcw_limits = limits.get("PCW", {})

    idx = pd.DatetimeIndex(res["index"])
    q = np.asarray(res["Q"], dtype=float)
    q_ratio = q / q_high if np.isfinite(q_high) and q_high > 0 else np.full(len(q), np.nan)

    pcw = np.asarray(res["PCW"], dtype=float)
    out_of_band = np.zeros(len(idx), dtype=float)
    if pcw.ndim == 2 and pcw_limits:
        for k in range(pcw.shape[1]):
            band = pcw_limits.get(f"PCW_{k + 1}") or pcw_limits.get(k + 1)
            if not band:
                continue
            out_of_band += ((pcw[:, k] > band["high"]) | (pcw[:, k] < band["low"])).astype(float)

    return idx, q_ratio, out_of_band


def train_and_score(
    ds: Dataset,
    protocol: str,
    clusters: list[dict],
    window: int = PCA_CLUSTER_WINDOW,
) -> list[str]:
    """Ajusta o baseline em todos os pares (cluster, condição) e grava no cache."""
    t0 = time.time()
    q_by_cluster: dict[str, pd.Series] = {}
    pcw_by_cluster: dict[str, pd.Series] = {}
    n_models = 0

    for cluster in clusters:
        q_parts, pcw_parts = [], []
        for _label, health, score in cluster_frames(ds, cluster, protocol, window):
            try:
                idx, q_ratio, out_of_band = _fit_one(health, score, window)
            except Exception:  # noqa: BLE001 — condição degenerada não invalida o cluster
                continue
            q_parts.append(pd.Series(q_ratio, index=idx))
            pcw_parts.append(pd.Series(out_of_band, index=idx))
            n_models += 1
        if q_parts:
            q_by_cluster[cluster["abbrev"]] = pd.concat(q_parts).sort_index()
            pcw_by_cluster[cluster["abbrev"]] = pd.concat(pcw_parts).sort_index()

    index = ds.index
    names = []
    for name, per_cluster, ucl_value, kind in (
        ("rmwpca_q", q_by_cluster, 1.0, "rmwpca_sc_q"),
        ("rmwpca_pcw", pcw_by_cluster, 1.5, "rmwpca_sc_pcw_2of3"),
    ):
        agg = aggregate_cluster_ratios(per_cluster, index)
        keep = np.isfinite(agg)
        save_scores(
            name,
            protocol,
            index[keep],
            agg[keep],
            np.full(keep.sum(), ucl_value),
            ds.oc.to_numpy()[keep],
            meta={
                "kind": kind,
                "window": window,
                "num_stds": NUM_STDS,
                "n_cluster_oc_models": n_models,
                "n_clusters_scored": len(per_cluster),
                "aggregation": "max sobre clusters",
                "train_seconds": round(time.time() - t0, 1),
            },
        )
        names.append(name)
    return names
