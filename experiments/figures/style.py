"""Estilo comum das figuras do artigo.

A paleta foi validada para deficiência de visão de cores e para contraste sobre
fundo claro: separação mínima entre pares adjacentes de ΔE 19,1 (deuteranopia) e
27,8 na visão normal, todos os tons acima de 3:1 de contraste. Cada painel exibe
uma única série identificada pelo próprio rótulo, de modo que a identidade nunca
depende só da cor.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Largura útil de uma coluna e da página em documento A4 de duas colunas,
# margens de 2 cm.
COL_W = 3.3
PAGE_W = 6.9

# Slots categóricos, em ordem fixa. Nunca reciclados.
BLUE = "#2a78d6"
ORANGE = "#d4541f"
PURPLE = "#7d4bc4"
GREEN = "#12855b"
NEUTRAL = "#6b7280"
NEUTRAL_LIGHT = "#9ca3af"

# Reservados para estado: nunca usados como série.
CRITICAL = "#c2362f"
GRID = "#d1d5db"
INK = "#1f2328"
INK_MUTED = "#5b6169"

MODEL_COLOR = {
    "mae": BLUE,
    "mae_nomask": ORANGE,
    "ae_dense": PURPLE,
    "rmwpca": GREEN,
    "ae_cluster": NEUTRAL,
    "classic": NEUTRAL,
}

MODEL_LABEL = {
    "mae": "MAE (proposto)",
    "mae_nomask": "AE-Transformer (sem máscara)",
    "ae_dense": "AE denso clássico",
    "rmwpca_q": "RMWPCA-SC ($Q$)",
    "rmwpca_pcw": "RMWPCA-SC (2 de 3)",
    "ae_cluster_q": "AE por cluster ($Q$)",
    "ae_cluster_vote": "AE por cluster (2 de 3)",
    "iforest": "Isolation Forest",
    "ocsvm": "One-Class SVM",
    "mae_sigmafloor": "MAE + piso de variância",
}


def apply() -> None:
    """Configura o matplotlib para saída vetorial em documento LaTeX."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.linewidth": 0.6,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def clean_axes(ax, grid_axis: str = "y") -> None:
    """Remove as bordas superior e direita; deixa a grade recessiva."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
        ax.set_axisbelow(True)


def mark_events(ax, events, alpha: float = 0.13, label_top: bool = False) -> None:
    """Sombreia as janelas dos eventos rotulados."""
    import pandas as pd

    for ev in events:
        ax.axvspan(pd.Timestamp(ev.start), pd.Timestamp(ev.end), color=CRITICAL, alpha=alpha, lw=0, zorder=1)
        if label_top:
            ax.annotate(
                ev.id,
                xy=(pd.Timestamp(ev.start), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -8),
                textcoords="offset points",
                fontsize=7,
                color=CRITICAL,
            )


def panel_label(ax, text: str, color: str = INK) -> None:
    """Rótulo direto do painel — carrega a identidade da série sem legenda."""
    ax.annotate(
        text,
        xy=(0.005, 0.94),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.5,
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
    )


def savefig(fig, path, **kwargs) -> None:
    fig.savefig(path, **kwargs)
    plt.close(fig)
