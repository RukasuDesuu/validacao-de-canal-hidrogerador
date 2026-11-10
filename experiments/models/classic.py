"""Baselines clássicos: Isolation Forest e One-Class SVM.

Ambos recebem exatamente o mesmo pré-processamento e o mesmo conjunto de treino.
No script original o One-Class SVM era ajustado sobre as primeiras 2000 janelas
enquanto o Isolation Forest via o conjunto inteiro, sem que isso fosse declarado
— a comparação não media a diferença entre os algoritmos.

A redução por PCA antes dos detectores retém 95% da variância e é ajustada só no
bloco de treino. Serve à equidade e ao custo: o One-Class SVM é quadrático no
número de amostras e opera aqui sobre 24x136 = 3264 dimensões achatadas.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from experiments.cache import save_scores
from experiments.data import Dataset
from experiments.models.common import build_windows, finalize

PCA_VARIANCE = 0.95
CONTAMINATION = 0.01
NU = 0.01


def _reduce(flat: np.ndarray, train_mask: np.ndarray, seed: int = 42) -> tuple[np.ndarray, int]:
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=seed)
    pca.fit(flat[train_mask])
    return pca.transform(flat), int(pca.n_components_)


def train_and_score(
    ds: Dataset,
    protocol: str,
    seed: int = 42,
    sigma_floor_mode: str = "none",
) -> list[str]:
    """Ajusta IF e OC-SVM sobre a mesma representação e grava os dois ScoreSets."""
    ws = build_windows(ds, protocol, sigma_floor_mode=sigma_floor_mode)
    flat = ws.flat()
    reduced, n_components = _reduce(flat, ws.fit_mask, seed)
    train = reduced[ws.fit_mask]

    names = []

    t0 = time.time()
    iforest = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=seed)
    iforest.fit(train)
    scores_if = -iforest.decision_function(reduced)
    t_if = time.time() - t0

    index, score, ucl, oc, extra = finalize(ws, scores_if)
    save_scores(
        "iforest",
        protocol,
        index,
        score,
        ucl,
        oc,
        meta={
            "kind": "isolation_forest",
            "seed": seed,
            "n_estimators": 100,
            "contamination": CONTAMINATION,
            "pca_components": n_components,
            "pca_variance": PCA_VARIANCE,
            "sigma_floor_mode": sigma_floor_mode,
            "train_seconds": round(t_if, 1),
            **extra,
        },
    )
    names.append("iforest")

    t0 = time.time()
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=NU)
    ocsvm.fit(train)
    scores_svm = -ocsvm.decision_function(reduced)
    t_svm = time.time() - t0

    index, score, ucl, oc, extra = finalize(ws, scores_svm)
    save_scores(
        "ocsvm",
        protocol,
        index,
        score,
        ucl,
        oc,
        meta={
            "kind": "one_class_svm",
            "kernel": "rbf",
            "nu": NU,
            "pca_components": n_components,
            "pca_variance": PCA_VARIANCE,
            "n_train_samples": int(ws.fit_mask.sum()),
            "sigma_floor_mode": sigma_floor_mode,
            "train_seconds": round(t_svm, 1),
            **extra,
        },
    )
    names.append("ocsvm")

    return names
