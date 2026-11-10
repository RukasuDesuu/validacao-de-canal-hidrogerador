"""Contribuição dos clusters de sensores para o erro de reconstrução, por evento.

Agregado por cluster de sintoma, e não por tag: um mesmo sintoma reúne até 32
tags, e a lista por tag fica ilegível e redundante.

O elemento central da figura é o contraste — para o evento de instrumentação, os
seis clusters da malha de medição elétrica, que são os que efetivamente
falharam, aparecem destacados na última posição, com contribuição desprezível.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from experiments.clusters import ELECTRICAL_ABBREVS, load_clusters_from_xlsx
from experiments.config import ARTICLE_DIR, CACHE_DIR
from experiments.figures import style
from experiments.ground_truth import EVENTS_BY_ID

TOP_N = 8


def build(events=("E1", "E2"), out_path=None):
    style.apply()
    clusters, _ = load_clusters_from_xlsx()

    frames = {}
    for event_id in events:
        path = CACHE_DIR / f"shap_cluster__{event_id}.parquet"
        tag_path = CACHE_DIR / f"shap__{event_id}.parquet"
        if path.exists() and tag_path.exists():
            frames[event_id] = (pd.read_parquet(path), pd.read_parquet(tag_path))
    if not frames:
        raise SystemExit("nenhum resultado de SHAP no cache; rode `python -m experiments.run_explain`")

    fig, axes = plt.subplots(1, len(frames), figsize=(style.PAGE_W, 2.5), constrained_layout=True)
    axes = [axes] if len(frames) == 1 else list(axes)

    for ax, (event_id, (bc, per_tag)) in zip(axes, frames.items()):
        bc = bc.reset_index(drop=True)
        top = bc.head(TOP_N)

        elec = bc[bc["abbrev"].isin(ELECTRICAL_ABBREVS)]
        ranks = sorted(int(i) + 1 for i in elec.index)
        elec_total = float(elec["median"].sum())

        labels = [r["abbrev"] for _, r in top.iterrows()]
        values = list(top["median"])
        colors = [style.BLUE if event_id == "E1" else style.ORANGE] * len(values)

        labels.append(f"malha elétrica (6 clusters, {ranks[0]}º–{ranks[-1]}º)")
        values.append(elec_total)
        colors.append(style.CRITICAL)

        y = range(len(values))[::-1]
        ax.barh(list(y), values, color=colors, height=0.68)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=6)
        ax.get_yticklabels()[-1].set_color(style.CRITICAL)

        ev = EVENTS_BY_ID[event_id]
        kind = "falha de processo" if ev.kind == "process_fault" else "canal degenerado"
        n = int(per_tag["n_windows"].iloc[0])
        ax.set_title(f"{event_id} — {kind} ({n} janelas)", fontsize=7.5)
        ax.set_xlabel("Contribuição para o erro")
        style.clean_axes(ax, grid_axis="x")

    out = out_path or ARTICLE_DIR / "fig_shap.pdf"
    style.savefig(fig, out)
    return out


def _short(name: str, limit: int = 30) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


if __name__ == "__main__":
    print(build())
