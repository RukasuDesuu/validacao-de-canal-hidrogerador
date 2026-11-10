"""AutoEncoder denso clássico sobre a janela achatada.

É o baseline que a tese do artigo precisa: sem ele, a afirmação de que o
mascaramento evita o *identity mapping* não tem contra-exemplo. A arquitetura
segue a do pipeline de referência — gargalo latente de dimensão 3, camadas
ocultas (64, 32).
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cache import save_scores
from experiments.config import AE_CLUSTER_HIDDEN, AE_CLUSTER_LATENT, MAE_BATCH, MAE_EPOCHS, MAE_LR
from experiments.data import Dataset
from experiments.models.common import build_windows, finalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)  # a máquina tem 4 núcleos; o padrão do torch usa 2


class DenseAE(nn.Module):
    """MLP simétrico com gargalo latente."""

    def __init__(self, n_input: int, hidden=AE_CLUSTER_HIDDEN, latent: int = AE_CLUSTER_LATENT) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        prev = n_input
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc.append(nn.Linear(prev, latent))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        prev = latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, n_input))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_and_score(
    ds: Dataset,
    protocol: str,
    seed: int = 42,
    epochs: int = MAE_EPOCHS,
    sigma_floor_mode: str = "none",
    model_name: str = "ae_dense",
) -> str:
    """Treina o AE denso e grava o ScoreSet no cache."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ws = build_windows(ds, protocol, sigma_floor_mode=sigma_floor_mode)
    flat = ws.flat()
    X_train = torch.tensor(flat[ws.fit_mask], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train), batch_size=MAE_BATCH, shuffle=True)

    model = DenseAE(flat.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=MAE_LR)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = torch.mean((model(batch) - batch) ** 2)
            loss.backward()
            optimizer.step()
    train_seconds = time.time() - t0

    model.eval()
    scores = np.empty(len(flat), dtype=float)
    all_X = torch.tensor(flat, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(all_X), 256):
            batch = all_X[start : start + 256].to(DEVICE)
            scores[start : start + len(batch)] = torch.mean((model(batch) - batch) ** 2, dim=1).cpu().numpy()

    index, score, ucl, oc, extra = finalize(ws, scores)
    save_scores(
        model_name,
        protocol,
        index,
        score,
        ucl,
        oc,
        meta={
            "kind": "ae_dense",
            "seed": seed,
            "epochs": epochs,
            "hidden": str(AE_CLUSTER_HIDDEN),
            "latent": AE_CLUSTER_LATENT,
            "sigma_floor_mode": sigma_floor_mode,
            "train_seconds": round(train_seconds, 1),
            **extra,
        },
    )
    return model_name
