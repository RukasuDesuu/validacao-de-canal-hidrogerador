"""Figura do conjunto de dados: potência, condição operacional e blocos.

Substitui a ausência de descrição de dados no artigo — o leitor vê de uma vez o
período coberto, o efeito do filtro de potência, a alternância de regime e onde
caem os blocos de treino e os eventos.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from experiments.config import ARTICLE_DIR, MIN_POWER_MW, POWER_TAG, PROTOCOLS
from experiments.data import apply_power_filter, assign_oc, load_raw
from experiments.figures import style
from experiments.ground_truth import EVENTS


def build(out_path=None):
    style.apply()
    raw = load_raw()
    filtered, stats = apply_power_filter(raw)
    oc = assign_oc(filtered)

    fig, axes = plt.subplots(
        3, 1, figsize=(style.PAGE_W, 3.2), sharex=True, height_ratios=[2.8, 0.55, 1.0], constrained_layout=True
    )

    # (a) potência ativa com o corte do filtro
    ax = axes[0]
    power = raw[POWER_TAG]
    ax.plot(power.index, power.to_numpy(), color=style.BLUE, lw=0.3, alpha=0.8)
    ax.axhline(MIN_POWER_MW, color=style.INK_MUTED, ls="--", lw=0.7)
    ax.annotate(
        f"corte em {MIN_POWER_MW:.0f} MW: {stats['frac_removed']:.1%} das horas removidas",
        xy=(0.02, MIN_POWER_MW),
        xycoords=("axes fraction", "data"),
        xytext=(0, -10),
        textcoords="offset points",
        fontsize=6.5,
        color=style.INK_MUTED,
    )
    ax.set_ylabel("Potência ativa (MW)")
    ax.set_ylim(-8, 210)
    style.clean_axes(ax)
    style.mark_events(ax, EVENTS, label_top=True)
    style.panel_label(ax, "(a)")

    # (b) condição operacional — rampa sequencial de uma única matiz
    ax = axes[1]
    labels = sorted(oc.unique())
    cmap = LinearSegmentedColormap.from_list("oc", ["#dce9f8", "#0f3f7a"])
    code = pd.Categorical(oc, categories=labels).codes
    colors = cmap(code / max(len(labels) - 1, 1))
    ax.vlines(oc.index, 0, 1, colors=colors, linewidth=0.35)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("(b) OC", rotation=0, ha="right", va="center", fontsize=7)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    # (c) blocos de treino de cada protocolo
    ax = axes[2]
    names = list(PROTOCOLS)
    for i, pid in enumerate(names):
        start, end = PROTOCOLS[pid]["train"]
        y = len(names) - i - 1
        ax.barh(y, pd.Timestamp(end) - pd.Timestamp(start), left=pd.Timestamp(start), height=0.42, color=style.BLUE, alpha=0.8)
        score = PROTOCOLS[pid]["score"]
        s0, s1 = (score if score else (oc.index.min(), oc.index.max()))
        ax.barh(y - 0.32, pd.Timestamp(s1) - pd.Timestamp(s0), left=pd.Timestamp(s0), height=0.14, color=style.INK_MUTED, alpha=0.55)
    ax.set_yticks([len(names) - i - 1 for i in range(len(names))])
    ax.set_yticklabels([f"{pid}  " for pid in names], fontsize=7)
    ax.set_ylim(-1.05, len(names) - 0.4)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    style.mark_events(ax, EVENTS)
    ax.annotate(
        "barra larga: ajuste do modelo e dos limiares   ·   barra estreita: janela pontuada",
        xy=(0.005, 0.06),
        xycoords="axes fraction",
        fontsize=6.5,
        color=style.INK_MUTED,
    )
    style.panel_label(ax, "(c)")

    axes[-1].set_xlabel("")
    out = out_path or ARTICLE_DIR / "fig_dataset.pdf"
    style.savefig(fig, out)
    return out, stats


if __name__ == "__main__":
    print(build()[0])
