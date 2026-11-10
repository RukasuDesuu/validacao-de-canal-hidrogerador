"""MWAutoEncoder por cluster x condição operacional — o AutoEncoder de produção.

É a implementação de referência do trabalho anterior (AutoEncoder por cluster
com rede bayesiana a jusante). Grava duas leituras, pelos mesmos motivos do
baseline de PCA: a razão ``Q/UCL_Q`` como score contínuo e o voto "2 de 3" sobre
as componentes latentes ``W_AE``, que é a regra de alarme real.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from experiments.cache import save_scores
from experiments.config import (
    AE_CLUSTER_EPOCHS,
    AE_CLUSTER_HIDDEN,
    AE_CLUSTER_LATENT,
    AE_CLUSTER_WINDOW,
    MAE_LR,
    NUM_STDS,
)
from experiments.data import Dataset
from experiments.models.cluster_common import aggregate_cluster_ratios, cluster_frames


def _fit_one(health: pd.DataFrame, score: pd.DataFrame, window: int, seed: int):
    """Ajusta um MWAutoEncoder numa condição e pontua a série daquela condição."""
    from fault_diagnosis.technique.autoencoder import MWAutoEncoder

    model = MWAutoEncoder(model_arch="mlp", hidden_layers=AE_CLUSTER_HIDDEN, use_scaler=True, wae_method="abs_ratio")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            health,
            window_size=window,
            latent_dim=AE_CLUSTER_LATENT,
            num_epochs=AE_CLUSTER_EPOCHS,
            learning_rate=MAE_LR,
            random_seed=seed,
        )
        res = model.transform_current_data(score)

    limits = model.limits or {}
    q_high = float(limits.get("Q", {}).get("high", np.nan))
    wae_limits = limits.get("W_AE", {})

    idx = pd.DatetimeIndex(res["index"])
    q = np.asarray(res["Q"], dtype=float)
    q_ratio = q / q_high if np.isfinite(q_high) and q_high > 0 else np.full(len(q), np.nan)

    wae = np.asarray(res["W_AE"], dtype=float)
    votes = np.zeros(len(idx), dtype=float)
    if wae.ndim == 2:
        for k in range(wae.shape[1]):
            band = wae_limits.get(f"W_AE_{k + 1}")
            if not band:
                continue
            votes += ((wae[:, k] > band["high"]) | (wae[:, k] < band["low"])).astype(float)

    return idx, q_ratio, votes


def train_and_score(
    ds: Dataset,
    protocol: str,
    clusters: list[dict],
    window: int = AE_CLUSTER_WINDOW,
    seed: int = 42,
    verbose: bool = True,
) -> list[str]:
    """Ajusta o AutoEncoder em todos os pares (cluster, condição) e grava no cache."""
    t0 = time.time()
    q_by_cluster: dict[str, pd.Series] = {}
    vote_by_cluster: dict[str, pd.Series] = {}
    n_models = 0

    for n, cluster in enumerate(clusters, 1):
        q_parts, vote_parts = [], []
        for _label, health, score in cluster_frames(ds, cluster, protocol, window):
            try:
                idx, q_ratio, votes = _fit_one(health, score, window, seed)
            except Exception:  # noqa: BLE001 — condição degenerada não invalida o cluster
                continue
            q_parts.append(pd.Series(q_ratio, index=idx))
            vote_parts.append(pd.Series(votes, index=idx))
            n_models += 1
        if q_parts:
            q_by_cluster[cluster["abbrev"]] = pd.concat(q_parts).sort_index()
            vote_by_cluster[cluster["abbrev"]] = pd.concat(vote_parts).sort_index()
        if verbose:
            print(f"  [{n}/{len(clusters)}] {cluster['abbrev']}: {n_models} modelos, {time.time() - t0:.0f}s", flush=True)

    index = ds.index
    names = []
    for name, per_cluster, ucl_value, kind in (
        ("ae_cluster_q", q_by_cluster, 1.0, "mw_autoencoder_q"),
        ("ae_cluster_vote", vote_by_cluster, 1.5, "mw_autoencoder_wae_2of3"),
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
                "latent_dim": AE_CLUSTER_LATENT,
                "hidden": str(AE_CLUSTER_HIDDEN),
                "epochs": AE_CLUSTER_EPOCHS,
                "num_stds": NUM_STDS,
                "seed": seed,
                "n_cluster_oc_models": n_models,
                "n_clusters_scored": len(per_cluster),
                "aggregation": "max sobre clusters",
                "train_seconds": round(time.time() - t0, 1),
            },
        )
        names.append(name)
    return names
