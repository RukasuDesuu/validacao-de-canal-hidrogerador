"""Estudo de caso da falha de instrumentação — a figura central do artigo.

Mostra, no mesmo eixo de tempo: o degrau permanente nas seis tags da malha de
medição elétrica; a resposta dos detectores; e a condição operacional, cuja
primeira reentrada em ``P1_Q2`` é o que finalmente disparou o alarme no pipeline
de produção, 18 dias depois do degrau.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from experiments.cache import load_scores
from experiments.clusters import ELECTRICAL_ABBREVS, cluster_by_abbrev, load_clusters_from_xlsx
from experiments.config import ARTICLE_DIR
from experiments.data import prepare
from experiments.ground_truth import EVENTS_BY_ID
from experiments.metrics import adaptive_flags
from experiments.figures import style

ZOOM = ("2026-04-20", "2026-06-07 21:00")
DETECTORS = [
    ("mae_m050_s42", style.BLUE, "MAE (proposto)"),
    ("rmwpca_q", style.GREEN, "RMWPCA-SC (produção)"),
]


def build(protocol: str = "P2", out_path=None):
    style.apply()
    ds = prepare()
    clusters, _ = load_clusters_from_xlsx()
    by_abbrev = cluster_by_abbrev(clusters)
    ev = EVENTS_BY_ID["E2"]

    lo, hi = pd.Timestamp(ZOOM[0]), pd.Timestamp(ZOOM[1])
    window = (ds.index >= lo) & (ds.index <= hi)

    fig, axes = plt.subplots(
        3, 1, figsize=(style.PAGE_W, 3.6), sharex=True, height_ratios=[1.5, 1.5, 0.4], constrained_layout=True
    )

    # (a) as seis tags elétricas, cruas
    ax = axes[0]
    for i, abbrev in enumerate(ELECTRICAL_ABBREVS):
        cluster = by_abbrev[abbrev]
        for tag in cluster["tags"]:
            if tag not in ds.df.columns:
                continue
            series = ds.df.loc[window, tag]
            ax.plot(series.index, series.to_numpy(), lw=0.7, alpha=0.85, color=style.BLUE if i < 3 else style.ORANGE)
    for k, step in enumerate(ev.meta["steps"]):
        ax.axvline(pd.Timestamp(step), color=style.INK_MUTED, ls=":", lw=0.8)
        ax.annotate(
            pd.Timestamp(step).strftime("%d/%m"),
            xy=(pd.Timestamp(step), 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(1.5, -7 - 8 * (k % 2)),
            textcoords="offset points",
            fontsize=6,
            color=style.INK_MUTED,
        )
    ax.set_ylabel("Valor (série tratada)")
    style.clean_axes(ax)
    style.panel_label(ax, "(a) correntes (azul) e tensões (laranja) de fase")
    ax.annotate(
        "excursão de três dias, seguida de novo patamar permanente",
        xy=(0.005, 0.06),
        xycoords="axes fraction",
        fontsize=6,
        color=style.INK_MUTED,
    )

    # (b) resposta dos detectores
    ax = axes[1]
    for i_det, (name, color, label) in enumerate(DETECTORS):
        try:
            ss = load_scores(name, protocol)
        except FileNotFoundError:
            continue
        sel = np.asarray((ss.index >= lo) & (ss.index <= hi))
        if not sel.any():
            continue
        # Série: razão sobre o limite fixo, que preserva a magnitude do desvio.
        # Sobreposto: o limite adaptativo, que é o ponto de operação da tabela e
        # que sobe ao longo do evento — o desvio persiste três semanas e acaba
        # sendo incorporado à própria referência.
        ratio, thr, flags = adaptive_flags(ss)
        ax.plot(ss.index[sel], ratio[sel], lw=0.9, color=color, label=label)
        ax.plot(ss.index[sel], thr[sel], lw=0.7, ls=":", color=color, alpha=0.75)
        hits = flags & sel & np.asarray(ss.index >= pd.Timestamp(ev.start))
        if hits.any():
            k = int(np.argmax(hits))
            first = ss.index[k]
            delay = (first - pd.Timestamp(ev.start)).total_seconds() / 3600
            # Deslocamentos alternados: com dois detectores as anotações caem
            # em pontos próximos e se sobrepõem.
            dx, dy = (10, 26) if i_det == 0 else (10, -26)
            ax.annotate(
                f"1º alarme: +{delay:.0f} h",
                xy=(first, ratio[k]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.5,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7),
            )
    ax.axhline(1.0, color=style.INK_MUTED, ls="--", lw=0.6, alpha=0.7)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_ylabel("Score / limite fixo")
    style.clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, ncols=2)
    ax.annotate(
        "linha pontilhada: limite adaptativo",
        xy=(0.005, 0.06),
        xycoords="axes fraction",
        fontsize=6,
        color=style.INK_MUTED,
    )

    # (c) condição operacional, destacando a reentrada que disparou a produção
    ax = axes[2]
    oc_win = ds.oc[window]
    labels = sorted(ds.oc.unique())
    cmap = LinearSegmentedColormap.from_list("oc", ["#dce9f8", "#0f3f7a"])
    code = pd.Categorical(oc_win, categories=labels).codes
    ax.vlines(oc_win.index, 0, 1, colors=cmap(code / max(len(labels) - 1, 1)), linewidth=0.7)
    alarm = pd.Timestamp(ev.meta["production_alarm"])
    ax.axvline(alarm, color=style.CRITICAL, lw=1.0)
    ax.annotate(
        f"1ª reentrada em {ev.meta['oc_at_alarm']}: alarme do pipeline de referência ({alarm.strftime('%d/%m %Hh')})",
        xy=(alarm, 1.0),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=style.CRITICAL,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("(c) OC", rotation=0, ha="right", va="center", fontsize=7)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    out = out_path or ARTICLE_DIR / "fig_case_cfa.pdf"
    style.savefig(fig, out)
    return out


if __name__ == "__main__":
    print(build())
