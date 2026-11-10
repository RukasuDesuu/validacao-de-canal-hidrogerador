"""Detectores avaliados na injeção, com o mesmo contrato dos do artigo.

Cada função recebe ``Z`` já padronizado (exceto ``channel_check``, que opera
sobre as unidades originais, a montante) e devolve um escore por amostra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pca_q(Z: np.ndarray, train: np.ndarray, var_keep: float = 0.95) -> np.ndarray:
    """Estatística Q (erro quadrático de reconstrução no subespaço residual)."""
    from sklearn.decomposition import PCA

    p = PCA(n_components=var_keep, svd_solver="full").fit(Z[train])
    R = Z - p.inverse_transform(p.transform(Z))
    return (R**2).sum(axis=1)


def ae_dense(Z: np.ndarray, train: np.ndarray, seed: int = 42, epochs: int = 60) -> np.ndarray:
    """Autocodificador denso; escore é o erro quadrático médio de reconstrução."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    d = Z.shape[1]
    latent = max(2, d // 4)
    net = nn.Sequential(
        nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, latent), nn.ReLU(),
        nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, d),
    )
    Xtr = torch.tensor(Z[train], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 256):
            b = Xtr[perm[i : i + 256]]
            opt.zero_grad()
            loss = lossf(net(b), b)
            loss.backward()
            opt.step()
    with torch.no_grad():
        R = net(torch.tensor(Z, dtype=torch.float32)).numpy() - Z
    return (R**2).mean(axis=1)


def iforest(Z: np.ndarray, train: np.ndarray, seed: int = 42) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    m = IsolationForest(random_state=seed, n_estimators=200).fit(Z[train])
    return -m.score_samples(Z)


def ocsvm(Z: np.ndarray, train: np.ndarray, max_fit: int = 4000, seed: int = 42) -> np.ndarray:
    from sklearn.svm import OneClassSVM

    idx = np.flatnonzero(train)
    if len(idx) > max_fit:  # OCSVM é O(n²); subamostrar mantém o custo viável
        idx = np.random.default_rng(seed).choice(idx, max_fit, replace=False)
    m = OneClassSVM(nu=0.05, gamma="scale").fit(Z[idx])
    return -m.decision_function(Z)


def channel_check(X: pd.DataFrame, train: np.ndarray, max_unique: int = 12) -> np.ndarray:
    """O critério de `models/sensor_check.py`, aqui sobre as unidades originais.

    Travamento por cardinalidade absoluta no ajuste; alarme quando um canal
    travado assume valor fora do suporte que exibiu ali. Invariante a escala.
    """
    from experiments.models.sensor_check import SUPPORT_RTOL

    V = X.to_numpy(dtype=float)
    score = np.zeros(len(V))
    for j in range(V.shape[1]):
        col_tr = V[train, j]
        sup = np.unique(col_tr[np.isfinite(col_tr)])
        if len(sup) > max_unique:
            continue
        d = np.abs(V[:, j][:, None] - sup[None, :]).min(axis=1)
        scale = np.maximum(np.abs(V[:, j]), 1.0)
        score += ((d / scale) > SUPPORT_RTOL).astype(float)
    return score


DETECTORS = {
    "pca_q": pca_q,
    "ae_dense": ae_dense,
    "iforest": iforest,
    "ocsvm": ocsvm,
}
