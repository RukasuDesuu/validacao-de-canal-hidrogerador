"""Geração das tabelas LaTeX do artigo a partir do cache.

Nenhum número da tabela é digitado no ``main.tex``: tudo sai daqui, de modo que
o texto e os resultados não podem divergir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.cache import available, load_scores
from experiments.config import (
    ARTICLE_DIR,
    CACHE_DIR,
    MIN_POWER_MW,
    PROTOCOLS,
    TRAIN_STRIDE,
    WINDOW,
)
from experiments.metrics import evaluate_all

# Ordem de apresentação; a primeira coluna da tabela segue esta lista.
MODEL_ORDER = [
    "mae_m050_s42",
    "mae_m000_s42",
    "ae_dense",
    "ae_cluster_q",
    "ae_cluster_vote",
    "rmwpca_q",
    "rmwpca_pcw",
    "iforest",
    "ocsvm",
    "mae_sigmafloor",
    "sensor_check",
]

PRETTY = {
    "mae_m050_s42": "MAE (proposto)",
    "mae_m000_s42": "AE-Transformer sem máscara",
    "mae_m025_s42": "MAE ($r=0{,}25$)",
    "mae_m075_s42": "MAE ($r=0{,}75$)",
    "ae_dense": "AE denso clássico",
    "ae_cluster_q": "AE por cluster ($Q$)",
    "ae_cluster_vote": "AE por cluster (2 de 3)",
    "rmwpca_q": "RMWPCA-SC ($Q$)",
    "rmwpca_pcw": "RMWPCA-SC (2 de 3)",
    "iforest": "Isolation Forest",
    "ocsvm": "One-Class SVM",
    "mae_sigmafloor": "MAE + piso de variância",
    "sensor_check": "Teste de canal (proposto)",
}


def collect(protocol: str, with_bootstrap: bool = True) -> pd.DataFrame:
    """Métricas de todos os modelos disponíveis no cache para um protocolo."""
    sets = []
    for model, proto in available():
        if proto != protocol:
            continue
        sets.append(load_scores(model, proto))
    if not sets:
        raise SystemExit(f"cache vazio para o protocolo {protocol}")
    return evaluate_all(sets, with_bootstrap=with_bootstrap)


def mae_seed_summary(protocol: str, mask: int = 50) -> pd.DataFrame:
    """Média e desvio entre sementes da configuração principal do MAE."""
    rows = [
        load_scores(m, p) for m, p in available() if p == protocol and m.startswith(f"mae_m{mask:03d}_s")
    ]
    if not rows:
        return pd.DataFrame()
    df = evaluate_all(rows, with_bootstrap=False)
    return df


def _br(text: str) -> str:
    """Vírgula decimal, no padrão pt-BR do documento."""
    return text.replace(".", "{,}")


def _pct(fraction: float, decimals: int = 1) -> str:
    """Percentual com vírgula decimal e o sinal escapado para o LaTeX."""
    return _br(f"{fraction * 100:.{decimals}f}") + r"\%"


def _fmt(value, spec: str = "{:.3f}", dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    if isinstance(value, (bool, np.bool_)):
        return "sim" if value else "não"
    return _br(spec.format(value))


def metrics_table(out_path=None) -> str:
    """Tabela principal: modelos x métricas, com blocos por protocolo."""
    blocks = []
    for protocol in PROTOCOLS:
        try:
            df = collect(protocol)
        except SystemExit:
            continue
        df = df.set_index("model")
        order = [m for m in MODEL_ORDER if m in df.index]
        order += [m for m in df.index if m not in order and not m.startswith("mae_m")]
        evaluates = PROTOCOLS[protocol]["evaluates"]

        n_cols = 1 + 2 * len(["E1", "E2"]) + 3
        lines = [rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{PROTOCOLS[protocol]['label']}}}}} \\"]
        for model in order:
            row = df.loc[model]
            cells = [PRETTY.get(model, model.replace("_", r"\_"))]
            for eid in ["E1", "E2"]:
                if eid in evaluates:
                    cells.append(_fmt(row.get(f"detected_roll_{eid}")))
                    cells.append(_fmt(row.get(f"delay_roll_{eid}_h"), "{:.0f}"))
                else:
                    cells += ["--", "--"]
            cells.append(_fmt(row.get("ar_health_roll"), "{:.3f}"))
            cells.append(_fmt(row.get("ap"), "{:.3f}"))
            cells.append(_fmt(row.get("ar_health"), "{:.2f}"))
            lines.append(" & ".join(cells) + r" \\")
        blocks.append("\n".join(lines))

    body = "\n\\midrule\n".join(blocks)

    tex = rf"""% Gerado por experiments/tables.py — não editar à mão.
\begin{{tabular}}{{lcccccc c}}
\toprule
 & \multicolumn{{2}}{{c}}{{E1 (processo)}} & \multicolumn{{2}}{{c}}{{E2 (sensor)}} & \multicolumn{{2}}{{c}}{{Desempenho}} & Limiar fixo \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-8}}
Modelo & det. & atraso (horas) & det. & atraso (horas) & $AR$ & $AP$ & $AR_{{\text{{fixo}}}}$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""
    out = out_path or ARTICLE_DIR / "tab_metrics.tex"
    out.write_text(tex, encoding="utf-8")
    return str(out)


def dataset_table(out_path=None) -> str:
    """Tabela descritiva do conjunto de dados."""
    from experiments.clusters import load_clusters_from_xlsx
    from experiments.data import prepare

    ds = prepare()
    clusters, fault_modes = load_clusters_from_xlsx()
    s = ds.filter_stats

    rows = [
        ("Período", f"{ds.index.min():%d/%m/%Y}--{ds.index.max():%d/%m/%Y}"),
        ("Amostragem", "horária"),
        ("Registros brutos", f"{s['n_raw']:,}".replace(",", ".")),
        (
            f"Após filtro ($\\geq$ {MIN_POWER_MW:.0f} MW)",
            f"{s['n_filtered']:,}".replace(",", ".") + f" ($-${_pct(s['frac_removed'])})",
        ),
        ("Segmentos contíguos", f"{ds.n_segments}"),
        ("Tags de processo", f"{ds.df.shape[1]}"),
        ("Clusters de sintoma", f"{len(clusters)}"),
        ("Modos de falha (FMEA)", f"{len(fault_modes)}"),
        ("Condições operacionais", f"{ds.oc.nunique()} ($4P \\times 3H$)"),
        ("Janela / passo", f"{WINDOW} h / 1 h (ajuste {TRAIN_STRIDE} h)"),
        ("Eventos rotulados", "2 (processo, instrumento)"),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows)
    tex = f"""% Gerado por experiments/tables.py — não editar à mão.
\\begin{{tabular}}{{@{{}}lr@{{}}}}
\\toprule
\\multicolumn{{2}}{{l}}{{Unidade geradora UG-A / UHE São Manoel}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
"""
    out = out_path or ARTICLE_DIR / "tab_dataset.tex"
    out.write_text(tex, encoding="utf-8")
    return str(out)


def dump_metrics(out_path=None) -> str:
    """Consolida todas as métricas num parquet para conferência."""
    frames = []
    for protocol in PROTOCOLS:
        try:
            frames.append(collect(protocol))
        except SystemExit:
            continue
    df = pd.concat(frames, ignore_index=True)
    out = out_path or CACHE_DIR / "metrics.parquet"
    df.to_parquet(out)
    return str(out)


if __name__ == "__main__":
    print(dataset_table())
    print(metrics_table())
    print(dump_metrics())
