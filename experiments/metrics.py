"""Métricas de detecção.

Com apenas dois eventos rotulados, nenhuma diferença entre modelos no nível de
evento é estatisticamente testável. As métricas aqui são descritivas por
construção, e a taxa de alarme é reportada com o denominador explícito
(``AR_health``) em vez de uma "taxa de falsos alarmes" sem definição.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.cache import ScoreSet
from experiments.config import (
    BOOTSTRAP_BLOCK_H,
    BOOTSTRAP_N,
    PERSIST_K,
    PERSIST_N,
    PROTOCOLS,
)
from experiments.ground_truth import (
    EVENTS_BY_ID,
    LABEL_FAULT,
    LABEL_GREY,
    LABEL_HEALTHY,
    build_labels,
)

DETECTION_TOLERANCE_H = 24

# Limiar adaptativo: janela de retaguarda usada para reestimar o limite, e a
# defasagem que a separa do instante avaliado.
ROLLING_WINDOW_D = 30
ROLLING_LAG_D = 1


def rolling_ucl(
    ss: ScoreSet,
    window_days: int = ROLLING_WINDOW_D,
    lag_days: int = ROLLING_LAG_D,
    num_stds: float = 3.0,
) -> np.ndarray:
    """Limite de controle reestimado sobre uma janela de retaguarda.

    Opera sobre a razão ``score/limite fixo``, e não sobre o escore bruto, para
    preservar o condicionamento por condição operacional já embutido no limite
    fixo: o limite adaptativo corrige a deriva temporal sem desfazer a correção
    de regime.

    Um limiar fixo estimado num bloco de saúde passado não transfere ao longo de
    meses: nesta unidade os escores de operação saudável crescem por deriva do
    processo, e o limite fixo passa a ser excedido quase sempre. O limite móvel é
    o ponto de operação que uma implantação real usaria.

    A janela é estritamente passada — nenhuma amostra do instante avaliado ou
    posterior entra no cálculo. O preço é conhecido e precisa ser declarado: um
    limite que acompanha a série acaba absorvendo qualquer desvio que persista
    por mais que ``window_days``, o mesmo compromisso que torna o PCA de produção
    cego a um degrau permanente.
    """
    s = pd.Series(ss.ratio, index=ss.index).sort_index()
    lag = pd.Timedelta(days=lag_days)
    win = pd.Timedelta(days=window_days)
    shifted = s.copy()
    shifted.index = shifted.index + lag
    roll = shifted.rolling(win, min_periods=48)
    mu = roll.mean().reindex(s.index, method="ffill")
    sd = roll.std(ddof=1).reindex(s.index, method="ffill")
    ucl = mu + num_stds * sd
    return ucl.reindex(ss.index).to_numpy()


def adaptive_flags(ss: ScoreSet, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ponto de operação adaptativo compartilhado por métricas e figuras.

    Returns:
        ``(razao, limiar, alarmes)`` — todos na escala da razão ``score/limite
        fixo``, com o filtro de persistência já aplicado aos alarmes.
    """
    thr = rolling_ucl(ss, **kwargs)
    ratio = ss.ratio
    flags = persistence_filter(np.asarray(ratio > thr) & np.isfinite(thr) & np.isfinite(ratio))
    return ratio, thr, flags


def persistence_filter(flags: np.ndarray, k: int = PERSIST_K, n: int = PERSIST_N) -> np.ndarray:
    """Alarme só se ``k`` das últimas ``n`` janelas estiverem acima do limiar.

    É o pós-filtro que a operação de fato aplicaria; sem ele, a taxa de alarme
    mede ruído de janela isolada.
    """
    if len(flags) < n:
        return np.zeros_like(flags, dtype=bool)
    kernel = np.ones(n, dtype=int)
    counts = np.convolve(flags.astype(int), kernel, mode="full")[: len(flags)]
    return counts >= k


def _event_window(event_id: str, tolerance_h: int = DETECTION_TOLERANCE_H) -> tuple[pd.Timestamp, pd.Timestamp]:
    ev = EVENTS_BY_ID[event_id]
    tol = pd.Timedelta(hours=tolerance_h)
    return pd.Timestamp(ev.start) - tol, pd.Timestamp(ev.end) + tol


def detection_for_event(ss: ScoreSet, event_id: str, flags: np.ndarray | None = None) -> dict:
    """Detecção, atraso e margem para um evento.

    ``delay_h`` é medido a partir do início real do evento — pode ser negativo se
    o modelo alarma dentro da tolerância anterior ao início rotulado.
    """
    ev = EVENTS_BY_ID[event_id]
    flags = ss.detected if flags is None else flags
    lo, hi = _event_window(event_id)
    in_window = (ss.index >= lo) & (ss.index <= hi)
    in_event = (ss.index >= pd.Timestamp(ev.start)) & (ss.index <= pd.Timestamp(ev.end))

    if not in_window.any():
        return {"evaluated": False, "detected": False, "delay_h": np.nan, "max_ratio": np.nan}

    hits = flags & in_window
    detected = bool(hits.any())
    delay = np.nan
    if detected:
        first = ss.index[np.argmax(hits)]
        delay = (first - pd.Timestamp(ev.start)).total_seconds() / 3600.0

    ratios = ss.ratio[in_event]
    max_ratio = float(np.nanmax(ratios)) if in_event.any() and np.isfinite(ratios).any() else np.nan

    return {"evaluated": True, "detected": detected, "delay_h": delay, "max_ratio": max_ratio}


def alarm_rate_healthy(ss: ScoreSet, exclude: tuple[str, str] | None, flags: np.ndarray | None = None) -> dict:
    """Taxa de alarme sobre horas saudáveis, fora do bloco de treino."""
    flags = ss.detected if flags is None else flags
    y = build_labels(ss.index)
    mask = (y == LABEL_HEALTHY).to_numpy()
    if exclude:
        mask &= ~np.asarray((ss.index >= pd.Timestamp(exclude[0])) & (ss.index <= pd.Timestamp(exclude[1])))
    n = int(mask.sum())
    return {
        "ar_health": float(flags[mask].mean()) if n else np.nan,
        "n_healthy_h": n,
    }


def average_precision(ss: ScoreSet) -> float:
    """AP sobre o rótulo horário, ignorando as horas cinza.

    Usa o score contínuo, portanto independe da escolha de limiar.
    """
    y = build_labels(ss.index).to_numpy()
    keep = y != LABEL_GREY
    score = ss.ratio
    keep &= np.isfinite(score)
    if keep.sum() == 0 or len(np.unique(y[keep])) < 2:
        return np.nan
    return float(average_precision_score((y[keep] == LABEL_FAULT).astype(int), score[keep]))


def _block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma réplica de bootstrap por blocos (preserva autocorrelação)."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block, 1), size=n_blocks)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def bootstrap_ci(
    ss: ScoreSet,
    exclude: tuple[str, str] | None,
    flags: np.ndarray | None = None,
    n_boot: int = BOOTSTRAP_N,
    block: int = BOOTSTRAP_BLOCK_H,
    seed: int = 42,
) -> dict:
    """IC de 95% para ``AR_health`` por bootstrap de blocos."""
    flags = ss.detected if flags is None else flags
    y = build_labels(ss.index)
    mask = (y == LABEL_HEALTHY).to_numpy()
    if exclude:
        mask &= ~np.asarray((ss.index >= pd.Timestamp(exclude[0])) & (ss.index <= pd.Timestamp(exclude[1])))
    vals = flags[mask].astype(float)
    if len(vals) < block:
        return {"ar_health_lo": np.nan, "ar_health_hi": np.nan}

    rng = np.random.default_rng(seed)
    reps = np.array([vals[_block_bootstrap_indices(len(vals), block, rng)].mean() for _ in range(n_boot)])
    return {
        "ar_health_lo": float(np.percentile(reps, 2.5)),
        "ar_health_hi": float(np.percentile(reps, 97.5)),
    }


def evaluate(ss: ScoreSet, with_bootstrap: bool = True) -> dict:
    """Linha completa de métricas para um (modelo, protocolo)."""
    proto = PROTOCOLS[ss.protocol]
    exclude = proto["train"]
    flags_raw = ss.detected
    flags_persist = persistence_filter(flags_raw)

    # Ponto de operação adaptativo: mesmo escore, limite reestimado em janela de
    # retaguarda. Persistência aplicada por cima, como faria a operação.
    _, _, flags_roll = adaptive_flags(ss)

    row: dict = {
        "model": ss.model,
        "protocol": ss.protocol,
        "n_windows": len(ss.index),
        **alarm_rate_healthy(ss, exclude, flags_raw),
        "ap": average_precision(ss),
    }
    row["ar_health_3of5"] = alarm_rate_healthy(ss, exclude, flags_persist)["ar_health"]
    row["ar_health_roll"] = alarm_rate_healthy(ss, exclude, flags_roll)["ar_health"]

    for eid in proto["evaluates"]:
        res = detection_for_event(ss, eid, flags_raw)
        row[f"detected_{eid}"] = res["detected"] if res["evaluated"] else np.nan
        row[f"delay_{eid}_h"] = res["delay_h"]
        row[f"max_ratio_{eid}"] = res["max_ratio"]
        res_roll = detection_for_event(ss, eid, flags_roll)
        row[f"detected_roll_{eid}"] = res_roll["detected"] if res_roll["evaluated"] else np.nan
        row[f"delay_roll_{eid}_h"] = res_roll["delay_h"]

    if with_bootstrap:
        row.update(bootstrap_ci(ss, exclude, flags_raw))

    row.update({f"meta_{k}": v for k, v in ss.meta.items() if isinstance(v, (int, float, str, bool))})
    return row


def evaluate_all(score_sets: list[ScoreSet], with_bootstrap: bool = True) -> pd.DataFrame:
    """Tabela de métricas para vários (modelo, protocolo)."""
    return pd.DataFrame([evaluate(ss, with_bootstrap) for ss in score_sets])
