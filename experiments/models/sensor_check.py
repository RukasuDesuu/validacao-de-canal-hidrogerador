"""Validação de canal a montante da normalização.

Motivação
---------
Um sensor travado num único valor durante todo o bloco de ajuste é
*inexpressável* em um modelo normalizado por escore padrão: não há dispersão
contra a qual medir seu desvio. O piso de variância avaliado por ablação
(``OCScaler(sigma_floor_mode="global")``) não corrige o caso, porque a
estatística global daquele canal é igualmente nula e o escalonador cai na guarda
numérica ``σ = 1``; o degrau de 0,70 vira um desvio de 0,70, contra excursões dez
vezes maiores em variáveis não relacionadas.

A saída não é um detector melhor. É um teste que **não dependa de dispersão**.

O critério
----------
Um sensor físico amostrado de hora em hora percorre um contínuo. Um canal que
assume 5 valores distintos em 13 mil horas não está medindo: está travado. Isso é
detectável **sem escala nenhuma** — e é exatamente o que falta a um modelo
normalizado por dispersão.

Dois passos, ambos invariantes a escala:

1. **Travamento** — o canal exibe no máximo ``max_unique`` valores distintos no
   bloco de ajuste. É um atributo do canal, não do instante, e é deliberadamente
   **graduado**: um teste de variância exatamente nula pegaria só o caso extremo
   e deixaria passar canais quase travados.

   O critério é a cardinalidade **absoluta**, não a razão
   ``n_distintos / n_amostras``. A razão acopla o limiar ao tamanho do bloco: os
   mesmos seis canais dão 1,9e-4 no bloco de P1 (5.175 h) e 3,8e-4 no de P2
   (7.932 h), e um limiar de 3e-4 marca seis canais em P1 e nenhum em P2. A
   cardinalidade absoluta não tem esse acoplamento, porque o número de valores de
   um canal travado cresce com o número de destravamentos, não com o tempo.
2. **Novidade** — o canal travado assume um valor fora do suporte que exibiu no
   ajuste. Para um canal que por definição não se move, mover-se já é o evento.

O escore de uma janela é a **contagem** de canais travados que exibiram valor
novo dentro dela. Não há limiar calibrado por dispersão, e essa ausência é o
ponto: a decisão é ``contagem >= 1``.

Observação sobre as unidades
---------------------------
As colunas do CSV têm prefixo ``DPP_TRATADA_`` e já chegam tratadas pelo PI
System — não são unidades de engenharia. O teste opera no ponto mais a montante
disponível neste conjunto: **antes da normalização por condição operacional**.
Como o critério é invariante a escala, isso não o enfraquece.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Cardinalidade maxima para o canal ser considerado travado. Qualquer valor em
# [4, 48] produz o mesmo conjunto nos dois protocolos: os seis canais travados
# exibem 1 valor no bloco de P1 e 3 no de P2, e o proximo canal — um sensor de
# entreferro, quantizado — exibe 49 em ambos. Ver `sweep`.
DEFAULT_MAX_UNIQUE = 12

# Tolerância relativa para pertencimento ao suporte. Os valores vêm do
# historiador em precisão simples; comparar por igualdade exata acusaria
# novidade a cada ruído de último bit.
SUPPORT_RTOL = 1e-6


@dataclass
class ChannelReport:
    """Estatísticas de um canal no bloco de ajuste."""

    tag: str
    n_unique: int
    n_samples: int
    modal_frac: float
    sigma: float
    support: np.ndarray

    @property
    def uniq_ratio(self) -> float:
        return self.n_unique / self.n_samples


@dataclass
class ChannelValidator:
    """Marca canais travados no ajuste e sinaliza quando eles se movem.

    Args:
        max_unique: cardinalidade máxima para o canal ser considerado travado.
    """

    max_unique: int = DEFAULT_MAX_UNIQUE
    reports_: dict[str, ChannelReport] = field(default_factory=dict)
    columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "ChannelValidator":
        self.columns_ = list(df.columns)
        n = len(df)
        for col in self.columns_:
            s = df[col].dropna()
            vc = s.value_counts()
            self.reports_[col] = ChannelReport(
                tag=col,
                n_unique=int(s.nunique()),
                n_samples=n,
                modal_frac=float(vc.iloc[0] / len(s)) if len(s) else np.nan,
                sigma=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                support=np.sort(s.unique()),
            )
        return self

    @property
    def latched(self) -> list[str]:
        """Canais travados no ajuste, em ordem crescente de cardinalidade."""
        out = [c for c in self.columns_ if self.reports_[c].n_unique <= self.max_unique]
        return sorted(out, key=lambda c: self.reports_[c].n_unique)

    def novelty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matriz booleana (hora x canal travado): valor fora do suporte de ajuste."""
        cols = self.latched
        out = pd.DataFrame(False, index=df.index, columns=cols)
        for col in cols:
            sup = self.reports_[col].support
            x = df[col].to_numpy(dtype=float)
            # distância ao valor de suporte mais próximo, em termos relativos
            d = np.abs(x[:, None] - sup[None, :]).min(axis=1)
            scale = np.maximum(np.abs(x), 1.0)
            out[col] = (d / scale) > SUPPORT_RTOL
        return out

    def score_hourly(self, df: pd.DataFrame) -> pd.Series:
        """Contagem de canais travados com valor novo, hora a hora."""
        nov = self.novelty(df)
        return nov.sum(axis=1).astype(float) if len(nov.columns) else pd.Series(0.0, index=df.index)


def score_windows(
    hourly: pd.Series,
    segments: np.ndarray,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Agrega o escore horário nas mesmas janelas dos modelos globais.

    O escore da janela é o **máximo** do escore horário dentro dela: quantos
    canais travados se moveram, no pior instante da janela. Usar o máximo — e não
    a soma — mantém a grandeza na mesma unidade do escore horário (número de
    canais) e não infla o valor por duração.
    """
    vals = hourly.to_numpy(dtype=float)
    starts = [
        s
        for s in range(0, len(vals) - window + 1, stride)
        if segments[s] == segments[s + window - 1]
    ]
    starts_arr = np.asarray(starts, dtype=int)
    if not len(starts_arr):
        return np.empty(0), np.empty(0, dtype=int)
    score = np.array([vals[s : s + window].max() for s in starts_arr])
    return score, starts_arr + window - 1


def sweep(
    validator_fit_df: pd.DataFrame,
    values: list[int] | None = None,
) -> pd.DataFrame:
    """Quantos canais o critério marca, em função de ``max_unique``.

    Serve para mostrar que a escolha não é ajuste fino: há uma faixa larga em que
    o conjunto marcado não muda.
    """
    values = values or [1, 2, 3, 4, 8, 12, 24, 48, 49, 64, 128]
    base = ChannelValidator().fit(validator_fit_df)
    rows = []
    for v in values:
        base.max_unique = v
        marked = base.latched
        rows.append({"max_unique": v, "n_latched": len(marked)})
    return pd.DataFrame(rows)


def train_and_score(ds, protocol: str, max_unique: int = DEFAULT_MAX_UNIQUE, model_name: str = "sensor_check") -> str:
    """Ajusta o validador no bloco do protocolo, pontua a série e grava no cache.

    "Treino" aqui é só a leitura das estatísticas de cardinalidade e do suporte
    de cada canal — não há parâmetro ajustado por otimização, e o custo é de
    segundos, não de horas.

    O limite de controle é a constante ``0,5``: qualquer canal travado que exiba
    valor novo já é alarme. Não há limiar calibrado por dispersão, porque é
    justamente a dispersão que falta ao canal. Manter o contrato ``score/ucl``
    permite que este detector entre nas mesmas tabelas e figuras dos demais.
    """
    import time

    from experiments.cache import save_scores
    from experiments.config import PROTOCOLS, STRIDE, WINDOW
    from experiments.models.common import protocol_bounds

    t0 = time.time()
    train_bounds, score_bounds = protocol_bounds(protocol)
    idx = ds.index
    train_rows = (idx >= pd.Timestamp(train_bounds[0])) & (idx <= pd.Timestamp(train_bounds[1]))

    validator = ChannelValidator(max_unique=max_unique).fit(ds.df.loc[train_rows])
    hourly = validator.score_hourly(ds.df)
    score, last_idx = score_windows(hourly, ds.segments, WINDOW, STRIDE)
    last_ts = idx[last_idx]

    if score_bounds is not None:
        keep = (last_ts >= pd.Timestamp(score_bounds[0])) & (last_ts <= pd.Timestamp(score_bounds[1]))
        score, last_ts = score[keep], last_ts[keep]

    ucl = np.full(len(score), 0.5)
    oc = ds.oc.reindex(last_ts).to_numpy().astype("U16")

    meta = {
        "kind": "sensor_check",
        "max_unique": max_unique,
        "n_latched": len(validator.latched),
        "latched_tags": [c for c in validator.latched],
        "ucl_rule": "constante 0.5 (contagem >= 1)",
        "n_scored_windows": int(len(score)),
        "train_seconds": round(time.time() - t0, 1),
    }
    save_scores(model_name, protocol, last_ts, score, ucl, oc, meta)
    return model_name
