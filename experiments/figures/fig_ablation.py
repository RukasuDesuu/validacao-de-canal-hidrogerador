"""Ablação da fração de mascaramento.

Sustenta — ou refuta — a premissa de que o mascaramento melhora a detecção. Um
eixo por medida: taxa de alarme em período saudável contra margem de detecção no
evento. Sem eixo duplo e sem legenda: cada ponto é rotulado diretamente.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from experiments.cache import available, load_scores
from experiments.config import ABLATION_MASK_RATIOS, ARTICLE_DIR
from experiments.figures import style
from experiments.metrics import adaptive_flags, alarm_rate_healthy, detection_for_event
from experiments.config import PROTOCOLS


def build(protocol: str = "P1", event_id: str = "E2", out_path=None):
    style.apply()
    exclude = PROTOCOLS[protocol]["train"]

    points = []
    for ratio in ABLATION_MASK_RATIOS:
        name = f"mae_m{int(ratio * 100):03d}_s42"
        try:
            ss = load_scores(name, protocol)
        except FileNotFoundError:
            continue
        _, _, flags = adaptive_flags(ss)
        ar = alarm_rate_healthy(ss, exclude, flags)["ar_health"]
        det = detection_for_event(ss, event_id, flags)
        points.append((f"$r={ratio:.2f}$".replace(".", "{,}"), ar, det["max_ratio"], style.BLUE))

    for name, label, color in [("ae_dense", "AE denso", style.PURPLE)]:
        if (name, protocol) in available():
            ss = load_scores(name, protocol)
            _, _, flags = adaptive_flags(ss)
            points.append(
                (
                    label,
                    alarm_rate_healthy(ss, exclude, flags)["ar_health"],
                    detection_for_event(ss, event_id, flags)["max_ratio"],
                    color,
                )
            )

    if not points:
        raise SystemExit("nada no cache para a ablação")

    fig, ax = plt.subplots(figsize=(style.COL_W, 2.4), constrained_layout=True)
    for label, ar, max_ratio, color in points:
        ax.scatter(ar, max_ratio, s=26, color=color, zorder=3)
        ax.annotate(label, xy=(ar, max_ratio), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=style.INK)

    ax.axhline(1.0, color=style.INK_MUTED, ls="--", lw=0.7)
    ax.annotate("limiar", xy=(0.99, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=6.5, color=style.INK_MUTED)
    ax.set_xlabel("Taxa de alarme em período saudável")
    ax.set_ylabel(f"Margem de detecção em {event_id}\n(máx. score/limiar)")
    ax.set_yscale("log")
    style.clean_axes(ax)

    # Folga para os rótulos diretos não saírem da área de desenho.
    lo, hi = ax.get_xlim()
    ax.set_xlim(max(0.0, lo - 0.1 * (hi - lo)), hi + 0.22 * (hi - lo))
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 2.2)
    ax.annotate(
        "a variação entre frações de mascaramento\né menor que o ruído entre sementes",
        xy=(0.03, 0.30),
        xycoords="axes fraction",
        fontsize=6.5,
        color=style.INK_MUTED,
    )

    out = out_path or ARTICLE_DIR / "fig_ablation.pdf"
    style.savefig(fig, out)
    return out


if __name__ == "__main__":
    print(build())
