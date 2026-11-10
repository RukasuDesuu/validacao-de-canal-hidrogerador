"""Rótulos dos eventos conhecidos da UG-A.

Não existem rótulos de campo neste dataset. Os dois eventos abaixo vêm dos
relatórios em ``docs/``, com as datas corrigidas contra os dados brutos:

* **E1 — falha de processo.** Entupimento dos filtros do trocador de calor,
  confirmado por intervenção física. Os relatórios registram 17/02/2025, mas o
  pico medido (z-score por OC e por cluster) está em 16/02, com TARE 3,57,
  TNE 3,79 e TEE 3,20 — cerca do p99 de cada cluster. É um desvio marginal, só
  visível sob normalização por condição operacional.

* **E2 — falha de instrumentação.** Degrau permanente nas seis tags da malha de
  medição elétrica. Os relatórios registram 19/05/2026, mas a série mostra três
  degraus: 16/05 00:00, 18/05 19:00 e 19/05 10:00, após 17 meses travada num
  valor constante. A corrente de fase A assume seis valores distintos em 18 meses.
  O pipeline de produção só alarmou em 03/06 16:00 — 18 dias depois — na primeira
  reentrada na condição operacional ``P1_Q2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LABEL_HEALTHY = 0
LABEL_FAULT = 1
LABEL_GREY = -1


@dataclass(frozen=True)
class Event:
    """Um evento rotulado."""

    id: str
    kind: str  # "process_fault" | "sensor_fault"
    start: str
    end: str
    grey: tuple[str, str] | None = None
    note: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def slice(self) -> slice:
        return slice(pd.Timestamp(self.start), pd.Timestamp(self.end))


EVENTS: list[Event] = [
    Event(
        id="E1",
        kind="process_fault",
        start="2025-02-16 00:00",
        end="2025-02-18 23:00",
        grey=("2025-02-10 00:00", "2025-02-22 23:00"),
        note="entupimento do trocador de calor; pico medido em 16/02 (relato: 17/02)",
    ),
    Event(
        id="E2",
        kind="sensor_fault",
        start="2026-05-16 00:00",
        end="2026-06-07 21:00",
        grey=None,
        note="degrau nas 6 tags elétricas; sensores permanecem defeituosos até o fim da série",
        meta={
            "steps": ["2026-05-16 00:00", "2026-05-18 19:00", "2026-05-19 10:00"],
            "production_alarm": "2026-06-03 16:00",
            "production_alarm_end": "2026-06-03 20:00",
            "oc_at_alarm": "P1_Q2",
        },
    ),
]

EVENTS_BY_ID = {e.id: e for e in EVENTS}


def build_labels(index: pd.DatetimeIndex, events: list[Event] | None = None) -> pd.Series:
    """Vetor de rótulos horário com 0 = saudável, 1 = falha, -1 = cinza (excluído).

    O halo cinza absorve a incerteza da data de início sem contaminar o
    denominador de alarmes em período saudável.
    """
    events = events if events is not None else EVENTS
    y = pd.Series(LABEL_HEALTHY, index=index, dtype=int)
    for ev in events:
        if ev.grey:
            y.loc[pd.Timestamp(ev.grey[0]) : pd.Timestamp(ev.grey[1])] = LABEL_GREY
    for ev in events:
        y.loc[ev.slice] = LABEL_FAULT
    return y


def healthy_mask(index: pd.DatetimeIndex, exclude: tuple[str, str] | None = None) -> pd.Series:
    """Horas contáveis como saudáveis, opcionalmente excluindo o bloco de treino.

    Excluir o bloco de treino é obrigatório: reportar taxa de alarme sobre dados
    usados para ajustar os limiares seria uma medida dentro da amostra.
    """
    y = build_labels(index)
    mask = y == LABEL_HEALTHY
    if exclude:
        mask &= ~((index >= pd.Timestamp(exclude[0])) & (index <= pd.Timestamp(exclude[1])))
    return pd.Series(mask, index=index)


def event_mask(index: pd.DatetimeIndex, event_id: str) -> np.ndarray:
    """Máscara booleana das horas de um evento."""
    ev = EVENTS_BY_ID[event_id]
    return ((index >= pd.Timestamp(ev.start)) & (index <= pd.Timestamp(ev.end))).to_numpy()
