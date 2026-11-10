"""Injeção controlada de defeito de canal em conjuntos públicos.

Pergunta
--------
O achado da UG-A é propriedade do *método* ou desta série? Com dois eventos
rotulados e uma única unidade, a série não decide. A resposta vem de reproduzir
o mecanismo num conjunto de terceiros, com defeito injetado sob controle.

O mecanismo, em uma linha
-------------------------
Um canal travado tem dispersão nula no ajuste. O escalonador cai na guarda
numérica ``σ = 1``, e o degrau de δ unidades vira um desvio padronizado de δ —
não de ``δ/σ``. Como as excursões normais dos demais canais são de ordem 1 a 3
em unidades padronizadas, um degrau com δ pequeno fica submerso, por maior que
seja em unidades de engenharia.

Desenho
-------
Três braços sobre o mesmo conjunto, mesmo canal-vítima, mesmo instante:

* ``clean``   — nada injetado. Mede a taxa de alarme de base.
* ``step``    — degrau de δ no canal-vítima, que mantém sua variância natural.
  **Braço de controle**: isola o efeito do degrau do efeito do travamento.
* ``latched`` — o canal-vítima é travado num valor constante durante todo o
  ajuste e o início do teste, e só então recebe o mesmo degrau de δ.

Se ``step`` detecta e ``latched`` não, a cegueira é causada pelo travamento, e
não pelo tamanho do degrau. É essa a afirmação a testar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    """Conjunto preparado, já dividido."""

    name: str
    X: pd.DataFrame          # todas as amostras, unidades originais
    train: np.ndarray        # máscara de ajuste
    cal: np.ndarray          # máscara de calibração do limiar (fora do ajuste)
    test: np.ndarray         # máscara de teste
    units: str = "engenharia"


def make_split(X: pd.DataFrame, frac_train: float = 0.5, frac_cal: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divisão estritamente temporal: ajuste, calibração, teste.

    A calibração fica **entre** ajuste e teste, e nunca entra no ajuste: um
    limiar estimado sobre os próprios dados de ajuste é excedido quase sempre
    fora da amostra.
    """
    n = len(X)
    i1 = int(n * frac_train)
    i2 = int(n * (frac_train + frac_cal))
    m = np.zeros(n, dtype=bool)
    train, cal, test = m.copy(), m.copy(), m.copy()
    train[:i1] = True
    cal[i1:i2] = True
    test[i2:] = True
    return train, cal, test


def inject(
    X: pd.DataFrame,
    victim: str,
    arm: str,
    delta: float,
    onset: int,
    latch_value: float | None = None,
) -> pd.DataFrame:
    """Devolve uma cópia de ``X`` com o braço aplicado ao canal-vítima.

    Args:
        arm: ``"clean"``, ``"step"`` ou ``"latched"``.
        delta: magnitude do degrau, em unidades originais do canal.
        onset: índice posicional a partir do qual o degrau vale.
        latch_value: valor de travamento; por omissão, a mediana do canal.
    """
    if arm not in {"clean", "step", "latched"}:
        raise ValueError(f"braço desconhecido: {arm}")
    out = X.copy()
    if arm == "clean":
        return out

    v = out[victim].to_numpy(dtype=float).copy()
    if arm == "latched":
        c = float(np.median(v)) if latch_value is None else latch_value
        v[:] = c
    v[onset:] = v[onset:] + delta
    out[victim] = v
    return out


def standardize(X: pd.DataFrame, train: np.ndarray, guard: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """z-score ajustado no bloco de treino, com a guarda numérica de produção.

    ``guard`` reproduz ``OCScaler``: quando σ é nulo, o divisor vira 1,0. É essa
    substituição — e não a normalização em si — que torna o canal travado
    inexpressável.
    """
    mu = X.to_numpy(dtype=float)[train].mean(axis=0)
    sd = X.to_numpy(dtype=float)[train].std(axis=0, ddof=1)
    sd_safe = np.where(sd > 1e-12, sd, guard)
    Z = (X.to_numpy(dtype=float) - mu) / sd_safe
    return Z, sd
