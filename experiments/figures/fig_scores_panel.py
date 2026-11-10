"""Painel comparativo de scores de anomalia, normalizados pelo limiar.

Substitui as quatro figuras independentes da versão anterior do artigo, cujos
eixos tinham escalas incomparáveis (o do PCA e o do MAE chegavam a 3500 por
causa de um único outlier de |z| = 393 em dados não filtrados).

Dividir o score pelo próprio limite de controle põe todos os modelos na mesma
escala: 1,0 é a fronteira de alarme, seja qual for a métrica de origem.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.cache import load_scores
from experiments.config import ARTICLE_DIR
from experiments.figures import style
from experiments.ground_truth import EVENTS
from experiments.metrics import adaptive_flags

DEFAULT_PANELS = [
    ("mae_m050_s42", style.BLUE, "MAE (proposto)"),
    ("mae_m000_s42", style.ORANGE, "AE-Transformer (sem máscara)"),
    ("ae_dense", style.PURPLE, "AE denso clássico"),
    ("rmwpca_q", style.GREEN, "RMWPCA-SC por cluster"),
    ("iforest", style.NEUTRAL, "Isolation Forest"),
    ("ocsvm", style.NEUTRAL, "One-Class SVM"),
]


def build(protocol: str = "P1", panels=None, out_path=None):
    style.apply()
    panels = panels or DEFAULT_PANELS

    available = []
    for name, color, label in panels:
        try:
            available.append((load_scores(name, protocol), color, label))
        except FileNotFoundError:
            continue
    if not available:
        raise SystemExit(f"nenhum modelo no cache para o protocolo {protocol}")

    fig, axes = plt.subplots(
        len(available), 1, figsize=(style.PAGE_W, 0.78 * len(available) + 0.5), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for i, (ax, (ss, color, label)) in enumerate(zip(axes, available)):
        # Série: razão entre o escore e o limite FIXO, que é uma referência
        # estável. A subida lenta dessa curva é a deriva do processo.
        # Sobreposta: o limite ADAPTATIVO, que acompanha a deriva — e que, por
        # isso mesmo, acaba absorvendo desvios longos. As duas referências
        # juntas são o compromisso discutido no texto.
        ratio, thr, _ = adaptive_flags(ss)
        ax.plot(ss.index, ratio, color=color, lw=0.45, alpha=0.9, zorder=3)
        ax.plot(ss.index, thr, color=style.INK_MUTED, lw=0.7, ls=":", alpha=0.9, zorder=4)
        ax.axhline(1.0, color=style.INK_MUTED, ls="--", lw=0.6, alpha=0.7)

        # Escala simétrica: os escores dos detectores clássicos assumem valores
        # negativos, que uma escala logarítmica descartaria.
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.set_ylabel("")
        style.clean_axes(ax)
        style.mark_events(ax, EVENTS)
        style.panel_label(ax, label, color=color)
        ax.tick_params(axis="y", labelsize=6)

    axes[0].annotate(
        "limiar fixo",
        xy=(0.30, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=6,
        color=style.INK_MUTED,
    )
    axes[0].annotate(
        "limiar adaptativo",
        xy=(0.62, 0.86),
        xycoords="axes fraction",
        fontsize=6,
        color=style.INK_MUTED,
    )
    for ev in EVENTS:
        axes[0].annotate(
            ev.id,
            xy=(pd.Timestamp(ev.start), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -7),
            textcoords="offset points",
            fontsize=6.5,
            color=style.CRITICAL,
        )
    fig.supylabel("Score normalizado pelo limite de controle", fontsize=8)

    out = out_path or ARTICLE_DIR / f"fig_scores_panel_{protocol.lower()}.pdf"
    style.savefig(fig, out)
    return out


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "P1"))
