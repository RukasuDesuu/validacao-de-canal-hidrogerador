"""Diagnóstico por rede bayesiana dos dois eventos, com e sem hipótese de falha de aquisição.

Reproduz o diagnóstico do pipeline de produção (CPT 0,9/0,1 derivada da FMEA, inferência
numérica marginal) sobre os conjuntos de clusters ativos em E1 e E2, e testa a extensão
proposta no artigo: um modo de falha extra, "falha da cadeia de aquisição", associado aos
clusters da malha de medição elétrica, e o teste de canal como evidência adicional.

Escreve ``cache/diagnosis.csv``. Não treina nada.

    python -m experiments.run_diagnosis
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.clusters import load_clusters_from_xlsx
from experiments.config import CACHE_DIR
from fault_diagnosis.technique.numerical_bayesian_network import NumericalBayesianNetwork

P_ASSOCIATED = 0.9  # valores do pipeline de produção
P_BACKGROUND = 0.1
ELECTRIC = ["CFA", "CFB", "CFC", "TFABG", "TFBCG", "TFCAG"]
# Clusters que o pipeline de produção ativou em cada pico (relatorio_resultados_finais.md §2).
EVIDENCE = {
    "E1": ["TARE", "TARS", "TEE"],
    "E2": ELECTRIC + ["TEE"],
}
INSTR_FM = "Falha da cadeia de aquisição"
CHANNEL_TEST = "Canal travado (teste de canal)"


def build_cpt(clusters, fault_modes, with_instrument: bool, with_channel_test: bool):
    names = [f"Cluster {c['id']}" for c in clusters]
    fm_names = [f"#{f['id']} {f['name']}" for f in fault_modes]
    cpt = np.full((len(fault_modes), len(clusters)), P_BACKGROUND)
    for c in clusters:
        for fm_id in c["fault_modes"]:
            cpt[fm_id - 1, c["id"] - 1] = P_ASSOCIATED
    df = pd.DataFrame(cpt, columns=names)
    if with_channel_test:
        df[CHANNEL_TEST] = P_BACKGROUND
    df.insert(0, "Modo de Falha", fm_names)
    if with_instrument:
        row = {"Modo de Falha": f"#{len(fm_names) + 1} {INSTR_FM}"}
        for c in clusters:
            row[f"Cluster {c['id']}"] = P_ASSOCIATED if c["abbrev"] in ELECTRIC else P_BACKGROUND
        if with_channel_test:
            row[CHANNEL_TEST] = P_ASSOCIATED
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def main() -> None:
    clusters, fault_modes = load_clusters_from_xlsx()
    by_abbrev = {c["abbrev"]: c["id"] for c in clusters}
    rows = []
    variants = {
        "producao": (False, False),
        "com_no_instrumental": (True, False),
        "com_no_e_teste_de_canal": (True, True),
    }
    for variant, (inst, chan) in variants.items():
        cpt = build_cpt(clusters, fault_modes, inst, chan)
        for mode in ("marginal", "categorical"):
            bn = NumericalBayesianNetwork(cpt, mode=mode, background_prob=P_BACKGROUND)
            for ev, abbrevs in EVIDENCE.items():
                active = [f"Cluster {by_abbrev[a]}" for a in abbrevs]
                if chan and ev == "E2":
                    active.append(CHANNEL_TEST)
                p = bn.calculate_probabilities(active, mode=mode)
                for fm, prob in zip(bn.failure_modes, p):
                    rows.append({"variant": variant, "mode": mode, "event": ev, "fault_mode": fm, "p": float(prob)})
    out = pd.DataFrame(rows)
    out.to_csv(CACHE_DIR / "diagnosis.csv", index=False)
    for (variant, mode, ev), g in out.groupby(["variant", "mode", "event"], sort=False):
        top = g.nlargest(4, "p")
        print(f"{variant:26s} {mode:11s} {ev}: " + " | ".join(f"{r.fault_mode[:38]} {r.p:.4f}" for r in top.itertuples()))


if __name__ == "__main__":
    main()
