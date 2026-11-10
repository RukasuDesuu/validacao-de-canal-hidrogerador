"""Carga, filtro de potência, condição operacional, segmentação e janelamento."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import (
    CSV_PATH,
    EXPECTED_N_FILTERED,
    EXPECTED_N_RAW,
    HEAD_TAG,
    MIN_POWER_MW,
    OC_KEY,
    OC_TAGS,
    POWER_TAG,
    STRIDE,
    TIME_COL,
    WINDOW,
)


def load_raw(csv_path: Path = CSV_PATH, validate: bool = True) -> pd.DataFrame:
    """Carrega o CSV com índice temporal."""
    df = pd.read_csv(csv_path, parse_dates=[TIME_COL]).set_index(TIME_COL).sort_index()
    if validate:
        assert len(df) == EXPECTED_N_RAW, f"{len(df)} linhas != {EXPECTED_N_RAW}"
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Colunas de processo (exclui as tags operacionais de potência e queda)."""
    return [c for c in df.columns if c not in OC_TAGS]


def apply_power_filter(
    df: pd.DataFrame,
    min_mw: float = MIN_POWER_MW,
    power_tag: str = POWER_TAG,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Remove horas de máquina desligada ou em baixa carga.

    Sem esse filtro os scores de anomalia são dominados por paradas de máquina:
    o máximo |z| do conjunto bruto chega a 393 contra um p99 de 8,9.
    """
    mask = df[power_tag] >= min_mw
    out = df.loc[mask]
    stats = {
        "n_raw": len(df),
        "n_filtered": len(out),
        "n_removed": len(df) - len(out),
        "frac_removed": (len(df) - len(out)) / len(df),
        "min_mw": min_mw,
    }
    if validate:
        assert len(out) == EXPECTED_N_FILTERED, f"{len(out)} linhas != {EXPECTED_N_FILTERED}"
    return out, stats


def assign_oc(df: pd.DataFrame, oc_key: str = OC_KEY) -> pd.Series:
    """Rótulo de condição operacional (P{1..4}_Q{1..3}) por amostra."""
    from fault_diagnosis.training.operational_condition_orchestrator import (
        OperationalConditionOrchestrator,
    )

    orch = OperationalConditionOrchestrator.from_json(oc_key)
    return orch.assign_conditions(df)


def oc_orchestrator(oc_key: str = OC_KEY):
    """Instância do orquestrador (para acessar limiares e rótulos)."""
    from fault_diagnosis.training.operational_condition_orchestrator import (
        OperationalConditionOrchestrator,
    )

    return OperationalConditionOrchestrator.from_json(oc_key)


def segment_ids(index: pd.DatetimeIndex, freq: str = "1h") -> np.ndarray:
    """Numera trechos de amostragem contígua.

    O filtro de potência abre buracos na série (o maior tem 20 dias). Janelas que
    atravessam um buraco misturariam regimes separados por semanas.
    """
    step = pd.Timedelta(freq)
    gaps = index.to_series().diff() != step
    return gaps.cumsum().to_numpy() - 1


def make_windows(
    values: np.ndarray,
    segments: np.ndarray,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Janelas deslizantes inteiramente contidas num segmento contíguo.

    Returns:
        ``(X, last_idx)`` com ``X`` de forma ``(n_windows, window, n_features)`` e
        ``last_idx`` o índice posicional da última amostra de cada janela.
    """
    starts = []
    for start in range(0, len(values) - window + 1, stride):
        end = start + window
        if segments[start] == segments[end - 1]:
            starts.append(start)
    starts_arr = np.asarray(starts, dtype=int)
    X = np.stack([values[s : s + window] for s in starts_arr]) if len(starts_arr) else np.empty((0, window, values.shape[1]))
    return X, starts_arr + window - 1


@dataclass
class Dataset:
    """Série preparada e seus metadados."""

    df: pd.DataFrame  # apenas colunas de sensor
    oc: pd.Series  # rótulo de OC por amostra
    power: pd.Series
    head: pd.Series
    segments: np.ndarray
    filter_stats: dict

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.df.index

    @property
    def n_segments(self) -> int:
        return int(self.segments.max()) + 1

    def windows(self, window: int = WINDOW, stride: int = STRIDE):
        """Janelas + timestamps do último ponto de cada janela."""
        X, last_idx = make_windows(self.df.to_numpy(), self.segments, window, stride)
        return X, self.index[last_idx]


def prepare(validate: bool = True) -> Dataset:
    """Pipeline completo de preparo: carga -> filtro -> OC -> segmentação."""
    raw = load_raw(validate=validate)
    filtered, stats = apply_power_filter(raw, validate=validate)
    oc = assign_oc(filtered)
    cols = sensor_columns(filtered)
    return Dataset(
        df=filtered[cols],
        oc=oc,
        power=filtered[POWER_TAG],
        head=filtered[HEAD_TAG],
        segments=segment_ids(filtered.index),
        filter_stats=stats,
    )
