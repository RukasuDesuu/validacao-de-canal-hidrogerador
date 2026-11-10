"""Figuras exclusivas da monografia (ilustrações didáticas e linha do tempo do evento).

Lê o CSV de sensores e ``cache/diagnosis.csv``; não treina nada. Os números de atribuição
(z-scores, conversão de unidades) são os já reportados no artigo e estão tabelados abaixo.
As figuras 'esquemáticas' usam sinais sintéticos declarados como tais na legenda do LaTeX.

    python -m experiments.figures.mono_figs
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.config import CACHE_DIR, CFA_TAG, MONOGRAFIA_DIR, ROOT
from experiments.figures import style

CFA = CFA_TAG
STEPS = ["2026-05-16 00:00", "2026-05-18 19:00", "2026-05-19 10:00"]
CHANNEL_ALARM = "2026-05-16 23:00"  # primeira hora pontuada após o degrau (filtro de potência)
PROD_ALARM = "2026-06-03 16:00"


def _save(fig, name):
    MONOGRAFIA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(MONOGRAFIA_DIR / name)
    plt.close(fig)
    print("ok", name)


def timeline():
    style.apply()
    s = pd.read_csv(ROOT / "dados_pi_extraidos.csv", usecols=["Data - Hora", CFA], parse_dates=["Data - Hora"]).set_index("Data - Hora")[CFA]
    w = s["2026-05-08":"2026-06-10"]
    fig, ax = plt.subplots(figsize=(style.PAGE_W, 2.8))
    ax.step(w.index, w.values, where="post", color=style.BLUE, lw=1.2)
    ymin, ymax = w.min(), w.max()
    for t in STEPS:
        ax.axvline(pd.Timestamp(t), color=style.NEUTRAL_LIGHT, lw=0.8, ls=":")
    ax.axvline(pd.Timestamp(CHANNEL_ALARM), color=style.GREEN, lw=1.4)
    ax.axvline(pd.Timestamp(PROD_ALARM), color=style.CRITICAL, lw=1.4)
    top = ymax + 0.55 * (ymax - ymin)
    ax.set_ylim(ymin - 0.15 * (ymax - ymin), top)
    ax.annotate("16/mai 23h: teste de canal alarma\n(primeira hora pontuada)", (pd.Timestamp(CHANNEL_ALARM), top), xytext=(-5, -4),
                textcoords="offset points", ha="right", va="top", color=style.GREEN, fontsize=8)
    ax.annotate("3/jun: alarme do pipeline de\nreferência, 17,7 dias depois", (pd.Timestamp(PROD_ALARM), top), xytext=(-5, -4),
                textcoords="offset points", ha="right", va="top", color=style.CRITICAL, fontsize=8)
    ax.text(pd.Timestamp("2026-05-08 12:00"), ymax - 0.06, "valor único por 17 meses", fontsize=8, color=style.INK_MUTED, va="top")
    ax.set_ylabel("Corrente de fase A (série tratada)")
    ax.set_xlabel("2026")
    ax.grid(axis="y", color=style.GRID, lw=0.5)
    fig.autofmt_xdate()
    _save(fig, "fig_mono_timeline.pdf")


def zscores():
    style.apply()
    labels = ["Vazão de óleo\n(cluster térmico)", "Temp. do ar de resfriamento\n(saída)", "Corrente/tensão de fase\n(canal degenerado)"]
    vals = [28.0, 21.5, 0.80]
    colors = [style.NEUTRAL, style.NEUTRAL, style.CRITICAL]
    fig, ax = plt.subplots(figsize=(style.PAGE_W, 2.3))
    ax.barh(labels, vals, color=colors, height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 60)
    for y, v in enumerate(vals):
        ax.text(v * 1.08, y, f"|z| = {v:g}".replace(".", ","), va="center", fontsize=8, color=style.INK)
    ax.set_xlabel("Desvio padronizado durante o evento E2 (escala logarítmica)")
    ax.grid(axis="x", color=style.GRID, lw=0.5)
    _save(fig, "fig_mono_zscores.pdf")


def units():
    style.apply()
    labels = ["kA", "A", "mA", "µA"]
    z = [0.00049, 0.487, 487, 487200]
    fig, ax = plt.subplots(figsize=(style.COL_W * 1.6, 2.4))
    ax.bar(labels, z, color=style.BLUE, width=0.55)
    ax.set_yscale("log")
    ax.axhline(1.0, color=style.CRITICAL, lw=1, ls="--")
    ax.text(-0.45, 1.35, "guarda $g = 1$", color=style.CRITICAL, fontsize=8, ha="left")
    for x, v in enumerate(z):
        ax.text(x, v * 1.4, f"{v:g}".replace(".", ","), ha="center", fontsize=8)
    ax.set_ylabel("$z$ do mesmo degrau físico")
    ax.set_xlabel("Unidade em que o sensor é expresso")
    ax.set_ylim(1e-4, 3e6)
    ax.grid(axis="y", color=style.GRID, lw=0.5)
    _save(fig, "fig_mono_unidades.pdf")


def pca_centering():
    style.apply()
    t = np.arange(-300, 300)
    x = np.where(t < 0, 1.21, 0.51).astype(float)
    W = 180
    fig, axes = plt.subplots(2, 3, figsize=(style.PAGE_W, 3.4), sharey="row")
    starts = [-200, -90, 20]
    titles = ["janela antes do degrau", "janela contém o degrau", "janela após o degrau"]
    for k, (a0, ttl) in enumerate(zip(starts, titles)):
        m = (t >= a0) & (t < a0 + W)
        xw = x[m]
        axes[0, k].plot(t, x, color=style.NEUTRAL_LIGHT, lw=0.8)
        axes[0, k].plot(t[m], xw, color=style.BLUE, lw=1.6)
        axes[0, k].set_title(ttl, fontsize=8)
        axes[1, k].plot(t[m], xw - xw.mean(), color=style.CRITICAL if k == 1 else style.ORANGE, lw=1.6)
        axes[1, k].axhline(0, color=style.GRID, lw=0.6)
        axes[1, k].set_xlabel("tempo (h, relativo ao degrau)")
    axes[0, 0].set_ylabel("sinal bruto")
    axes[1, 0].set_ylabel("sinal centrado\npela média da janela")
    for k in (0, 2):
        axes[1, k].text(0.5, 0.15, "vetor nulo", transform=axes[1, k].transAxes, ha="center", fontsize=8, color=style.INK_MUTED)
    for a in axes.ravel():
        a.grid(axis="y", color=style.GRID, lw=0.5)
    fig.tight_layout()
    _save(fig, "fig_mono_pca_autocentramento.pdf")


def diagnosis():
    style.apply()
    d = pd.read_csv(CACHE_DIR / "diagnosis.csv")
    d = d[d["mode"] == "categorical"]
    short = lambda s: {"Falha da cadeia de aquisição": "Falha de aquisição"}.get(s.split(" ", 1)[1], s.split(" ", 1)[1].split(" (")[0])
    panels = [("E1", "producao", "E1 (processo), produção"), ("E2", "producao", "E2 (canal degenerado), produção"),
              ("E2", "com_no_instrumental", "E2, com nó de falha de aquisição"), ("E2", "com_no_e_teste_de_canal", "E2, com nó e teste de canal")]
    fig, axes = plt.subplots(2, 2, figsize=(style.PAGE_W, 4.0), sharex=True)
    for ax, (ev, var, ttl) in zip(axes.ravel(), panels):
        g = d[(d.event == ev) & (d.variant == var)].nlargest(4, "p").iloc[::-1]
        cols = [style.CRITICAL if "aquisição" in f else style.NEUTRAL for f in g.fault_mode]
        ys = np.arange(len(g))
        ax.barh(ys, g.p, color=cols, height=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{short(f)} ({f.split(' ')[0]})" for f in g.fault_mode])
        for y, v in enumerate(g.p):
            ax.text(v + 0.02, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=7)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", color=style.GRID, lw=0.5)
    fig.supxlabel("Probabilidade do modo de falha (inferência categórica)", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_mono_diagnostico.pdf")



# Participação por fonte na geração elétrica do Brasil (% do total anual), agregada a partir de
# Ember, Yearly Electricity Data (https://ember-energy.org/data/yearly-electricity-data/),
# arquivo release_generation_yearly_global.csv, acessado em 04/11/2026. "Fóssil" soma gás natural,
# carvão e outras fontes fósseis; "Bio/nuclear" soma bioenergia e geração nuclear.
BRAZIL_MIX = {
    2000: (87.24, 0.00, 4.30, 3.98), 2001: (81.53, 0.01, 6.39, 7.09), 2002: (82.75, 0.02, 6.23, 6.97),
    2003: (83.87, 0.02, 6.07, 6.94), 2004: (82.79, 0.02, 7.65, 6.23), 2005: (83.72, 0.02, 7.32, 5.83),
    2006: (83.16, 0.06, 6.85, 6.82), 2007: (84.02, 0.15, 5.74, 6.83), 2008: (79.79, 0.26, 8.79, 7.29),
    2009: (83.88, 0.27, 5.00, 7.69), 2010: (78.19, 0.42, 9.26, 9.01), 2011: (80.55, 0.51, 7.09, 9.08),
    2012: (75.17, 0.91, 11.08, 9.38), 2013: (68.50, 1.15, 15.94, 9.88), 2014: (63.24, 2.07, 18.28, 10.58),
    2015: (61.89, 3.73, 18.32, 11.12), 2016: (65.80, 5.80, 14.19, 11.61), 2017: (62.94, 7.33, 15.01, 11.65),
    2018: (64.68, 8.63, 12.54, 11.65), 2019: (62.82, 9.89, 13.84, 11.22), 2020: (63.04, 10.78, 12.28, 11.58),
    2021: (55.27, 13.56, 16.94, 10.77), 2022: (63.07, 16.50, 8.35, 10.28), 2023: (60.16, 20.68, 7.33, 10.20),
    2024: (56.14, 23.73, 8.24, 10.28), 2025: (52.10, 26.69, 9.82, 10.14),
}  # ano -> (hidro, eólica+solar, fóssil, bio+nuclear), em % da geração


def brazil_matrix():
    style.apply()
    anos = sorted(BRAZIL_MIX)
    hidro, renov, fossil, outras = (np.array([BRAZIL_MIX[a][i] for a in anos]) for i in range(4))
    fig, ax = plt.subplots(figsize=(style.PAGE_W, 3.2))
    series = [("Hidrelétrica", hidro, style.BLUE, 0), ("Eólica + solar", renov, style.GREEN, 0),
              ("Fóssil (gás, carvão)", fossil, style.NEUTRAL, 6), ("Bioenergia + nuclear", outras, style.PURPLE, -6)]
    for nome, vals, cor, dy in series:
        ax.plot(anos, vals, color=cor, lw=1.4)
        ax.annotate(nome, (anos[-1], vals[-1]), xytext=(4, dy), textcoords="offset points",
                    ha="left", va="center", color=cor, fontsize=7)
    ax.set_ylabel("Participação na geração elétrica (\\%)")
    ax.set_xlabel("Ano")
    ax.set_xlim(anos[0], anos[-1] + 6)
    ax.set_ylim(0, 95)
    ax.grid(axis="y", color=style.GRID, lw=0.5)
    fig.tight_layout()
    _save(fig, "fig_mono_matriz.pdf")


if __name__ == "__main__":
    timeline(); zscores(); units(); pca_centering(); diagnosis(); brazil_matrix()
