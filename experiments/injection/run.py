"""Executa a injeção controlada e tabula o resultado.

Uso::

    python -m experiments.injection.run --dataset skab
    python -m experiments.injection.run --dataset smd --victim-rank 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.injection import detectors as det
from experiments.injection.harness import inject, make_split, standardize

DATA = Path(__file__).resolve().parents[2] / "data_public"
OUT = Path(__file__).resolve().parents[2] / "cache" / "injection"

# Magnitudes do degrau, em múltiplos do desvio-padrão natural do canal-vítima.
# Expressar em múltiplos de σ torna a grade comparável entre conjuntos e entre
# canais; a conversão para unidades de engenharia fica na tabela de saída.
DELTAS_SIGMA = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
NUM_STDS = 3.0


def load_skab() -> pd.DataFrame:
    df = pd.read_csv(DATA / "skab" / "anomaly-free.csv", sep=";", parse_dates=["datetime"])
    return df.drop(columns=["datetime"]).astype(float)


def load_smd() -> pd.DataFrame:
    m = np.loadtxt(DATA / "smd" / "machine-1-1-train.txt", delimiter=",")
    return pd.DataFrame(m, columns=[f"d{j:02d}" for j in range(m.shape[1])])


LOADERS = {"skab": load_skab, "smd": load_smd}


def evaluate_arm(X: pd.DataFrame, victim: str, arm: str, delta: float, onset: int,
                 train: np.ndarray, cal: np.ndarray, test: np.ndarray) -> list[dict]:
    """Roda todos os detectores num braço e devolve uma linha por detector."""
    Xi = inject(X, victim, arm, delta, onset)
    Z, sd = standardize(Xi, train)
    y = np.zeros(len(Xi), dtype=int)
    y[onset:] = 1

    # Atribuição: contribuição de cada canal ao desvio padronizado durante o
    # evento. É a grandeza que a camada de explicabilidade ordena, e a que decide
    # para onde a manutenção é enviada.
    j = list(Xi.columns).index(victim)
    post_all = np.arange(len(Xi)) >= onset
    contrib = (Z[post_all] ** 2).mean(axis=0)
    rank = int(np.where(np.argsort(-contrib) == j)[0][0]) + 1

    rows = []
    scored = {name: fn(Z, train) for name, fn in det.DETECTORS.items()}
    scored["channel_check"] = det.channel_check(Xi, train)

    for name, s in scored.items():
        ucl = s[cal].mean() + NUM_STDS * s[cal].std(ddof=1)
        if name == "channel_check":
            ucl = 0.5  # contagem >= 1; nao ha dispersao a calibrar
        flags = s > ucl
        pre = test & (np.arange(len(y)) < onset)
        post = test & (np.arange(len(y)) >= onset)
        rows.append({
            "arm": arm,
            "detector": name,
            "delta": delta,
            "ap": float(average_precision_score(y[test], s[test])) if len(np.unique(y[test])) > 1 else np.nan,
            "recall": float(flags[post].mean()) if post.any() else np.nan,
            "fpr": float(flags[pre].mean()) if pre.any() else np.nan,
            "sigma_victim_train": float(sd[j]),
            "victim_rank": rank,
            "n_channels": int(Xi.shape[1]),
            "victim_contrib": float(contrib[j]),
            "top_contrib": float(contrib.max()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    ap.add_argument("--victim", default=None, help="canal-vítima (padrão: o de maior desvio relativo)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = LOADERS[args.dataset]()
    train, cal, test = make_split(X)
    onset = int(len(X) * 0.75)  # metade do bloco de teste

    if args.victim is None:
        # canal contínuo mais "bem-comportado": maior cardinalidade no ajuste,
        # para que o braço `latched` seja de fato uma degradação e não o estado
        # natural do canal.
        card = X.iloc[train].nunique()
        args.victim = str(card.idxmax())

    sigma_nat = float(X.iloc[train][args.victim].std(ddof=1))
    print(f"conjunto={args.dataset}  n={len(X)}  canais={X.shape[1]}")
    print(f"vitima={args.victim}  sigma_natural={sigma_nat:.4g}  onset={onset}\n", flush=True)

    rows = []
    for d_sig in DELTAS_SIGMA:
        delta = d_sig * sigma_nat
        for arm in ("step", "latched"):
            for r in evaluate_arm(X, args.victim, arm, delta, onset, train, cal, test):
                r["delta_sigma"] = d_sig
                rows.append(r)
        print(f"  delta={d_sig}sigma ok", flush=True)
    for r in evaluate_arm(X, args.victim, "clean", 0.0, onset, train, cal, test):
        r["delta_sigma"] = 0.0
        rows.append(r)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{args.dataset}.csv", index=False)
    (OUT / f"{args.dataset}_meta.json").write_text(json.dumps(
        {"dataset": args.dataset, "victim": args.victim, "sigma_natural": sigma_nat,
         "n": len(X), "n_channels": int(X.shape[1]), "onset": onset,
         "deltas_sigma": DELTAS_SIGMA, "num_stds": NUM_STDS}, indent=2))
    print(f"\n-> {OUT / f'{args.dataset}.csv'}")


if __name__ == "__main__":
    main()
