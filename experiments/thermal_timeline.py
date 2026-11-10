"""Evolução semanal do desvio térmico (cluster TARS) e elétrico, jan a jun de 2026.

Responde "a anomalia térmica que domina o erro durante E2 já existia antes do defeito?".
Usa a mesma normalização por condição operacional do pipeline (protocolo P1), então os |z|
são os que o modelo enxerga. Imprime a tabela semanal que o texto cita; não treina nada.
"""

from __future__ import annotations

import pandas as pd

from experiments.clusters import load_clusters_from_xlsx
from experiments.config import PROTOCOLS
from experiments.data import prepare
from experiments.oc_scaler import OCScaler

ELEC = ["CFA", "CFB", "CFC", "TFABG", "TFBCG", "TFCAG"]


def main() -> None:
    ds = prepare(validate=False)
    clusters, _ = load_clusters_from_xlsx()
    by = {c["abbrev"]: c for c in clusters}
    a, b = PROTOCOLS["P1"]["train"]
    train = ds.df.loc[a:b]
    scaler = OCScaler("none").fit(train, ds.oc.loc[train.index])
    z = scaler.transform(ds.df, ds.oc).abs()
    tars = [t for t in by["TARS"]["tags"] if t in z.columns]
    elec = [t for c in ELEC for t in by[c]["tags"] if t in z.columns]
    w = z.loc["2026-01-01":"2026-06-10"]
    weekly = pd.DataFrame(
        {
            "TARS_absz": w[tars].mean(axis=1).resample("W").mean(),
            "eletricos_absz": w[elec].mean(axis=1).resample("W").mean(),
            "horas": w[tars[0]].resample("W").count(),
        }
    )
    print(weekly.round(2).to_string())
    print(f"\nbloco de saúde de {a[:4]} (P1): TARS |z| médio = {z.loc[a:b, tars].values.mean():.2f}")


if __name__ == "__main__":
    main()
