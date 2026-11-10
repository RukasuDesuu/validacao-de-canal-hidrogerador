"""Tabela de atribuição do evento E2 por detector (lê apenas de ``cache/attrib_*``).

Alimentada por ``run_attribution.py``. Com ``--com-classicos``, inclui o Isolation Forest e o
One-Class SVM (atribuídos por Shapley por tag) e grava só na pasta da monografia; sem a
flag, o comportamento é o do artigo. Para cada detector responde: onde está o erro (ou o
alarme) durante E2 e em que posição ficam os seis clusters da malha elétrica, que são os que
de fato falharam. Escreve ``tab_atribuicao.tex`` na pasta do artigo (e na da monografia, se o
``config`` a define) e imprime os números que vão para o RESULTADOS.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.clusters import load_clusters_from_xlsx
from experiments import config
from experiments.config import ARTICLE_DIR, CACHE_DIR
from experiments.ground_truth import EVENTS_BY_ID

CLASSIC_MODELS = [
    ("iforest", "Isolation Forest"),
    ("ocsvm", "One-Class SVM"),
]
ELEC = ["CFA", "CFB", "CFC", "TFABG", "TFBCG", "TFCAG"]
GLOBAL_MODELS = [
    ("mae_m050", "MAE global ($r = 0{,}50$)"),
    ("ae_transformer_sem_mascara", "AE-Transformer sem máscara"),
    ("ae_denso", "AutoEncoder denso"),
]


def _pt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", "{,}")


def _positions(pos: list[int]) -> str:
    pos = sorted(pos)
    runs, start = [], pos[0]
    for a, b in zip(pos, pos[1:] + [None]):
        if b is None or b != a + 1:
            runs.append(str(start) if start == a else (f"{start}, {a}" if a == start + 1 else f"{start}--{a}"))
            start = b
    return ", ".join(runs)


def global_row(model: str, clusters: list[dict], what: str = "do erro") -> dict:
    t = pd.read_parquet(CACHE_DIR / f"attrib_global__{model}__E2.parquet")
    rows = []
    for c in clusters:
        tg = [x for x in c["tags"] if x in t.index]
        if tg:
            rows.append((c["abbrev"], float(t.loc[tg, "median"].sum())))
    d = pd.DataFrame(rows, columns=["ab", "v"]).sort_values("v", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    d["share"] = d["v"] / d["v"].sum()
    e = d[d["ab"].isin(ELEC)]
    return {
        "lider": d.loc[0, "ab"],
        "lider_val": f"{_pt(100 * d.loc[0, 'share'], 0)}\\% {what}",
        "pos": _positions(list(e["rank"])),
        "elet": (f"{_pt(100 * e['share'].sum(), 2)}" if 100 * e['share'].sum() >= 0.005 else "$<0{,}01$") + f"\\% {what}",
    }


def cluster_row(model: str) -> dict:
    f = pd.read_parquet(CACHE_DIR / f"attrib_cluster__{model}__P1.parquet")
    ev = EVENTS_BY_ID["E2"]
    w = f.loc[ev.start : ev.end]
    st = pd.DataFrame({"max": w.max(), "frac": (w > 1).mean(), "n": w.notna().sum()})
    scored = st[st["n"] > 0].sort_values("max", ascending=False)
    ranks = {c: i + 1 for i, c in enumerate(scored.index)}
    elec_pos = [ranks[c] for c in ELEC if c in ranks]
    top = scored.index[0]
    return {
        "lider": top,
        "max": float(scored.loc[top, "max"]),
        "frac": float(scored.loc[top, "frac"]),
        "pos": _positions(elec_pos) if elec_pos else None,
        "n_scored": len(scored),
    }


def _labels(index: pd.DatetimeIndex, evs: list[str]) -> np.ndarray:
    y = np.zeros(len(index), bool)
    for e in evs:
        ev = EVENTS_BY_ID[e]
        y |= (index >= pd.Timestamp(ev.start)) & (index <= pd.Timestamp(ev.end))
    return y


def ap_with_without_electrical(model: str) -> dict:
    f = pd.read_parquet(CACHE_DIR / f"attrib_cluster__{model}__P1.parquet")
    out = {}
    for name, cols in (("todos", list(f.columns)), ("sem_eletricos", [c for c in f.columns if c not in ELEC])):
        s = f[cols].max(axis=1)
        ok = s.notna().to_numpy()
        y = _labels(f.index, ["E1", "E2"])[ok]
        out[name] = float(average_precision_score(y, s.to_numpy()[ok]))
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--com-classicos", action="store_true", help="inclui IF e OC-SVM (só monografia)")
    args = ap.parse_args()
    clusters, _ = load_clusters_from_xlsx()
    lines = []
    summary = {}
    for model, label in GLOBAL_MODELS:
        r = global_row(model, clusters)
        summary[model] = r
        lines.append(
            f"{label} & erro por \\textit{{tag}} & {r['lider']} ({r['lider_val']}) & {r['pos']} ({r['elet']}) \\\\"
        )
    classic_lines = []
    if args.com_classicos:
        for model, label in CLASSIC_MODELS:
            r = global_row(model, clusters, what="da atribuição")
            summary[model] = r
            classic_lines.append(
                f"{label} & Shapley por \\textit{{tag}} & {r['lider']} ({r['lider_val']}) & {r['pos']} ({r['elet']}) \\\\"
            )
    pca = cluster_row("rmwpca")
    ae = cluster_row("ae_cluster")
    summary["rmwpca"], summary["ae_cluster"] = pca, ae
    lines.append(
        f"PCA robusto de produção & razão $Q/\\mathrm{{UCL}}$ por \\textit{{cluster}} & {pca['lider']} "
        f"(razão máx. {_pt(pca['max'], 0)}; {_pt(100 * pca['frac'], 0)}\\% das horas) & não pontuados \\\\"
    )
    exp = int(np.floor(np.log10(ae["max"])))
    mant = ae["max"] / 10**exp
    lines.append(
        f"AE por \\textit{{cluster}} de produção & razão $Q/\\mathrm{{UCL}}$ por \\textit{{cluster}} & {ae['lider']} "
        f"(razão máx. ${_pt(mant, 1)}\\times10^{{{exp}}}$; {_pt(100 * ae['frac'], 0)}\\% das horas) & {ae['pos']} \\\\"
    )
    lines = lines[:3] + classic_lines + lines[3:] if classic_lines else lines
    tex = (
        "% Gerado por experiments/attribution_table.py a partir de cache/attrib_*; não editar à mão.\n"
        "\\begin{tabular}{@{}p{3.3cm}p{3.0cm}p{4.4cm}p{3.6cm}@{}}\n\\toprule\n"
        "Detector & Medida & Onde está o erro / alarme & Posição dos 6 \\textit{clusters} elétricos \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    if args.com_classicos:
        dirs = [config.MONOGRAFIA_DIR]  # a medição de IF/OC-SVM é só da monografia
    else:
        dirs = [ARTICLE_DIR] + ([config.MONOGRAFIA_DIR] if hasattr(config, "MONOGRAFIA_DIR") else [])
    for d in dirs:
        out = d / "tab_atribuicao.tex"
        out.write_text(tex, encoding="utf8")
        print(f"escrito: {out}")
    print(tex)

    print("\n== números para o RESULTADOS.md")
    for k, v in summary.items():
        print(k, v)
    for m in ("ae_cluster", "rmwpca"):
        print(m, "AP[E1+E2]:", {k: round(v, 3) for k, v in ap_with_without_electrical(m).items()})


if __name__ == "__main__":
    main()
