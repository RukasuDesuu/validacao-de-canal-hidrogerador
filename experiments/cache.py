"""Cache em disco dos scores de anomalia.

Contrato único entre a etapa de computar e a de desenhar: um artefato por
(modelo, protocolo). Nenhum script de figura ou tabela treina qualquer coisa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import CACHE_DIR


@dataclass
class ScoreSet:
    """Série de scores de um modelo sob um protocolo.

    Attributes:
        index: timestamp do último ponto de cada janela.
        score: score bruto de anomalia.
        ucl: limite de controle aplicável àquela janela (por condição operacional).
        oc: rótulo de condição operacional da janela.
        meta: hiperparâmetros, semente, tamanho do treino, tempo de execução.
    """

    model: str
    protocol: str
    index: pd.DatetimeIndex
    score: np.ndarray
    ucl: np.ndarray
    oc: np.ndarray
    meta: dict

    @property
    def ratio(self) -> np.ndarray:
        """``score / ucl``: 1,0 é o limiar, independentemente da escala do modelo."""
        return self.score / np.where(self.ucl > 0, self.ucl, np.nan)

    @property
    def detected(self) -> np.ndarray:
        return self.score > self.ucl

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"score": self.score, "ucl": self.ucl, "ratio": self.ratio, "oc": self.oc, "detected": self.detected},
            index=self.index,
        )


def _paths(model: str, protocol: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model}__{protocol}"
    return CACHE_DIR / f"scores__{stem}.npz", CACHE_DIR / f"meta__{stem}.json"


def save_scores(
    model: str,
    protocol: str,
    index: pd.DatetimeIndex,
    score: np.ndarray,
    ucl: np.ndarray,
    oc: np.ndarray,
    meta: dict,
) -> Path:
    """Grava um ScoreSet no cache."""
    npz_path, meta_path = _paths(model, protocol)
    np.savez_compressed(
        npz_path,
        index=np.asarray(index, dtype="datetime64[ns]"),
        score=np.asarray(score, dtype=float),
        ucl=np.asarray(ucl, dtype=float),
        oc=np.asarray(oc, dtype="U16"),
    )
    meta_path.write_text(json.dumps({"model": model, "protocol": protocol, **meta}, indent=2, default=str))
    return npz_path


def load_scores(model: str, protocol: str) -> ScoreSet:
    """Lê um ScoreSet do cache."""
    npz_path, meta_path = _paths(model, protocol)
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache ausente: {npz_path}. Rode `python -m experiments.run_all`.")
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ScoreSet(
        model=model,
        protocol=protocol,
        index=pd.DatetimeIndex(z["index"]),
        score=z["score"],
        ucl=z["ucl"],
        oc=z["oc"],
        meta=meta,
    )


def available() -> list[tuple[str, str]]:
    """Pares (modelo, protocolo) presentes no cache."""
    if not CACHE_DIR.exists():
        return []
    out = []
    for p in sorted(CACHE_DIR.glob("scores__*.npz")):
        stem = p.stem.removeprefix("scores__")
        model, _, protocol = stem.rpartition("__")
        out.append((model, protocol))
    return out


def exists(model: str, protocol: str) -> bool:
    return _paths(model, protocol)[0].exists()
