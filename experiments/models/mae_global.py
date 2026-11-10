"""Masked AutoEncoder global condicionado por condição operacional.

O mascaramento ocorre **apenas no treino** (``TimeSeriesMAE.forward`` verifica
``self.training``); na inferência a janela entra completa e o score é o erro
quadrático médio de reconstrução. O argumento contra o *identity mapping* vale,
portanto, para a representação aprendida, não para o modo de inferência — e é
por isso que a ablação sobre ``mask_ratio`` faz parte dos resultados.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.cache import save_scores
from experiments.config import (
    MAE_BATCH,
    MAE_D_MODEL,
    MAE_EPOCHS,
    MAE_LAYERS,
    MAE_LR,
    MAE_NHEAD,
    MASK_RATIO,
)
from experiments.data import Dataset
from experiments.models.common import build_windows, finalize
from mae_model import TimeSeriesMAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)  # a máquina tem 4 núcleos; o padrão do torch usa 2


def _loss(x: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE nas posições mascaradas; sobre tudo quando não há máscara.

    Com ``mask_ratio=0`` o denominador da perda mascarada seria zero — a variante
    sem máscara precisa reconstruir a janela inteira para treinar.
    """
    sq = (recon - x) ** 2
    denom = mask.sum()
    if denom.item() == 0:
        return sq.mean()
    return (sq.mean(dim=-1) * mask).sum() / denom


def train_and_score(
    ds: Dataset,
    protocol: str,
    mask_ratio: float = MASK_RATIO,
    seed: int = 42,
    epochs: int = MAE_EPOCHS,
    sigma_floor_mode: str = "none",
    model_name: str | None = None,
) -> str:
    """Treina o MAE no bloco do protocolo e grava o ScoreSet no cache."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ws = build_windows(ds, protocol, sigma_floor_mode=sigma_floor_mode)
    X_train = torch.tensor(ws.X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train), batch_size=MAE_BATCH, shuffle=True)

    model = TimeSeriesMAE(
        num_features=ws.n_features,
        d_model=MAE_D_MODEL,
        num_layers=MAE_LAYERS,
        nhead=MAE_NHEAD,
        mask_ratio=mask_ratio,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=MAE_LR)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mask = model(batch)
            loss = _loss(batch, recon, mask)
            loss.backward()
            optimizer.step()
    train_seconds = time.time() - t0

    model.eval()
    scores = np.empty(len(ws.X), dtype=float)
    all_X = torch.tensor(ws.X, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(all_X), 256):
            batch = all_X[start : start + 256].to(DEVICE)
            recon, _ = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=(1, 2))
            scores[start : start + len(batch)] = mse.cpu().numpy()

    index, score, ucl, oc, extra = finalize(ws, scores)
    name = model_name or f"mae_m{int(mask_ratio * 100):03d}_s{seed}"
    save_scores(
        name,
        protocol,
        index,
        score,
        ucl,
        oc,
        meta={
            "kind": "mae_global",
            "mask_ratio": mask_ratio,
            "seed": seed,
            "epochs": epochs,
            "d_model": MAE_D_MODEL,
            "num_layers": MAE_LAYERS,
            "nhead": MAE_NHEAD,
            "sigma_floor_mode": sigma_floor_mode,
            "train_seconds": round(train_seconds, 1),
            "device": str(DEVICE),
            **extra,
        },
    )
    return name


def torch_model_for_shap(ds: Dataset, protocol: str, mask_ratio: float = MASK_RATIO, seed: int = 42, epochs: int = MAE_EPOCHS):
    """Treina e devolve ``(modelo, WindowSet)`` para a análise de explicabilidade."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    ws = build_windows(ds, protocol)
    X_train = torch.tensor(ws.X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train), batch_size=MAE_BATCH, shuffle=True)

    model = TimeSeriesMAE(
        num_features=ws.n_features, d_model=MAE_D_MODEL, num_layers=MAE_LAYERS, nhead=MAE_NHEAD, mask_ratio=mask_ratio
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=MAE_LR)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon, mask = model(batch)
            _loss(batch, recon, mask).backward()
            optimizer.step()
    model.eval()
    return model, ws
