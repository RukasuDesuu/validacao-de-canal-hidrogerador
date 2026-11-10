"""Confere cada afirmação numérica do artigo contra os dados.

Roda depois de qualquer alteração no pipeline ou no texto. Toda linha impressa
com ``FALHA`` indica um número do ``main.tex`` que deixou de corresponder ao que
os dados dizem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.clusters import ELECTRICAL_ABBREVS, cluster_by_abbrev, load_clusters_from_xlsx
from experiments.config import PROTOCOLS
from experiments.data import prepare
from experiments.ground_truth import EVENTS_BY_ID
from experiments.oc_scaler import OCScaler

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((ok, name, detail))


def main() -> int:
    ds = prepare()
    clusters, fault_modes = load_clusters_from_xlsx()
    by_abbrev = cluster_by_abbrev(clusters)
    s = ds.filter_stats

    # --- Seção 3.1: conjunto de dados ---
    check("13.141 registros brutos", s["n_raw"] == 13141, str(s["n_raw"]))
    check("10.933 após o filtro", s["n_filtered"] == 10933, str(s["n_filtered"]))
    check("16,8% removidos", abs(s["frac_removed"] - 0.168) < 0.001, f"{s['frac_removed']:.4f}")
    check("136 tags de processo", ds.df.shape[1] == 136, str(ds.df.shape[1]))
    check("30 segmentos contíguos", ds.n_segments == 30, str(ds.n_segments))
    check("33 clusters", len(clusters) == 33, str(len(clusters)))
    check("13 modos de falha", len(fault_modes) == 13, str(len(fault_modes)))
    check("12 condições operacionais", ds.oc.nunique() == 12, str(ds.oc.nunique()))

    # --- Seção 3.2: faixas de operação ---
    check("potência máxima de 191 MW", abs(ds.power.max() - 191.0) < 1.5, f"{ds.power.max():.1f}")
    check("queda entre 14,2 e 31,4 m", ds.head.min() > 13 and ds.head.max() < 33, f"{ds.head.min():.1f}–{ds.head.max():.1f}")

    # --- Seção 4.3: as seis tags são elétricas ---
    esperado = {
        "CFA": "Corrente Fase A", "CFB": "Corrente Fase B", "CFC": "Corrente Fase C",
        "TFABG": "Tensão Fase AB", "TFBCG": "Tensão Fase BC", "TFCAG": "Tensão Fase CA",
    }
    for abbrev, nome in esperado.items():
        real = by_abbrev[abbrev]["name"]
        check(f"{abbrev} = {nome}", real == nome, real)

    # --- Seção 4.3: variância degenerada ---
    train = PROTOCOLS["P1"]["train"]
    tr = (ds.index >= pd.Timestamp(train[0])) & (ds.index <= pd.Timestamp(train[1]))
    rep = OCScaler("none").fit(ds.df.loc[tr], ds.oc).degenerate_report()
    tags_eletricas = {t for a in ELECTRICAL_ABBREVS for t in by_abbrev[a]["tags"]}
    check("72 pares (condição, tag) com desvio nulo", len(rep) == 72, str(len(rep)))
    check("as degeneradas são exatamente as 6 elétricas", set(rep.tag) == tags_eletricas, f"{rep.tag.nunique()} tags")
    check("em todas as 12 condições", rep.groupby("tag").size().eq(12).all(), str(rep.groupby("tag").size().unique()))
    check("1632 combinações no total", 12 * ds.df.shape[1] == 1632, str(12 * ds.df.shape[1]))

    # --- Seção 4.3: os degraus do sensor ---
    cfa = [t for t in by_abbrev["CFA"]["tags"] if t in ds.df.columns][0]
    ev = EVENTS_BY_ID["E2"]
    esperados = [pd.Timestamp(x) for x in ev.meta["steps"]]

    # A série bruta contém três degraus; o do dia 18/05 cai numa hora de baixa
    # carga, removida pelo filtro de potência, e portanto não aparece no
    # conjunto analisado.
    from experiments.data import load_raw

    bruto = load_raw()[cfa]
    degraus_brutos = bruto.diff().abs()
    degraus_brutos = degraus_brutos[degraus_brutos > 0.01].index
    check(
        "três degraus na série bruta",
        list(degraus_brutos[degraus_brutos >= "2026-01-01"]) == esperados,
        str(list(degraus_brutos[degraus_brutos >= "2026-01-01"])),
    )

    raw = ds.df[cfa]
    saltos = raw.diff().abs()
    degraus = saltos[saltos > 0.01].index
    check(
        "dois degraus sobrevivem ao filtro de potência",
        len(degraus[degraus >= "2026-01-01"]) == 2,
        str(list(degraus[degraus >= "2026-01-01"])),
    )
    check("degrau inicial em 16/05/2026", esperados[0] == pd.Timestamp("2026-05-16 00:00"), str(esperados[0]))
    n_uniq = raw.nunique()
    check("CFA com poucos valores distintos", n_uniq <= 6, f"{n_uniq} valores")

    antes = raw[raw.index < esperados[0]]
    travado_desde = antes[antes != antes.iloc[-1]].index.max()
    meses = (esperados[0] - travado_desde).days / 30.44
    check("travado por ~17 meses", 16 <= meses <= 18, f"{meses:.1f} meses")

    # --- Seção 4.3: atraso do alarme de produção ---
    dias = (pd.Timestamp(ev.meta["production_alarm"]) - esperados[0]).days
    check("18 dias entre o degrau e o alarme de produção", dias == 18, f"{dias} dias")

    # --- Seção 3.6: deriva concentrada ---
    sc = OCScaler("none").fit(ds.df.loc[tr], ds.oc)
    Z = sc.transform(ds.df, ds.oc)
    m26 = (ds.index > pd.Timestamp(train[1])) & (ds.index < pd.Timestamp(ev.start))
    contrib = (Z[m26] ** 2).mean().sort_values(ascending=False)
    frac12 = contrib.head(12).sum() / contrib.sum()
    check("12 tags respondem por ~85% do erro de 2026", 0.80 <= frac12 <= 0.90, f"{frac12:.1%}")

    largura = max(len(n) for _, n, _ in _results)
    falhas = 0
    for ok, name, detail in _results:
        status = "ok   " if ok else "FALHA"
        if not ok:
            falhas += 1
        print(f"[{status}] {name:<{largura}}  {detail}")
    print(f"\n{len(_results) - falhas}/{len(_results)} afirmações conferem")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
