"""Orquestra o treino e a pontuação de todos os modelos.

Uso::

    python -m experiments.run_all --model mae
    python -m experiments.run_all --model ae_cluster --protocol P1
    python -m experiments.run_all --all

Cada modelo escreve seu próprio artefato em ``cache/``; nenhum depende do outro,
de modo que rodar em processos separados é seguro.
"""

from __future__ import annotations

import argparse
import time

from experiments.clusters import load_clusters_from_xlsx
from experiments.config import ABLATION_MASK_RATIOS, MASK_RATIO, PROTOCOLS, SEEDS
from experiments.data import prepare

MODELS = ["mae", "mae_ablation", "ae_dense", "classic", "pca_cluster", "ae_cluster", "sigma_floor", "sensor_check"]


def run(model: str, protocols: list[str]) -> None:
    ds = prepare()
    print(f"dados: {len(ds.df)} horas, {ds.df.shape[1]} sensores, {ds.n_segments} segmentos", flush=True)

    for protocol in protocols:
        t0 = time.time()
        print(f"\n=== {model} / {protocol} ===", flush=True)

        if model == "mae":
            from experiments.models import mae_global

            for seed in SEEDS:
                name = mae_global.train_and_score(ds, protocol, mask_ratio=MASK_RATIO, seed=seed)
                print(f"  {name} ok", flush=True)

        elif model == "mae_ablation":
            from experiments.models import mae_global

            for ratio in ABLATION_MASK_RATIOS:
                name = mae_global.train_and_score(ds, protocol, mask_ratio=ratio, seed=SEEDS[0])
                print(f"  {name} ok", flush=True)

        elif model == "ae_dense":
            from experiments.models import ae_dense

            print(f"  {ae_dense.train_and_score(ds, protocol)} ok", flush=True)

        elif model == "classic":
            from experiments.models import classic

            print(f"  {classic.train_and_score(ds, protocol)} ok", flush=True)

        elif model == "pca_cluster":
            from experiments.models import pca_cluster

            clusters, _ = load_clusters_from_xlsx()
            print(f"  {pca_cluster.train_and_score(ds, protocol, clusters)} ok", flush=True)

        elif model == "ae_cluster":
            from experiments.models import ae_cluster

            clusters, _ = load_clusters_from_xlsx()
            print(f"  {ae_cluster.train_and_score(ds, protocol, clusters)} ok", flush=True)

        elif model == "sigma_floor":
            # Ablação da camada de sanidade de sensores: o mesmo MAE com piso de
            # variância por condição operacional.
            from experiments.models import mae_global

            name = mae_global.train_and_score(
                ds,
                protocol,
                mask_ratio=MASK_RATIO,
                seed=SEEDS[0],
                sigma_floor_mode="global",
                model_name="mae_sigmafloor",
            )
            print(f"  {name} ok", flush=True)

        elif model == "sensor_check":
            # Validação de canal a montante da normalização. Não treina nada:
            # lê cardinalidade e suporte de cada canal e pontua em segundos.
            from experiments.models import sensor_check

            print(f"  {sensor_check.train_and_score(ds, protocol)} ok", flush=True)

        else:
            raise SystemExit(f"modelo desconhecido: {model}")

        print(f"=== {model} / {protocol}: {time.time() - t0:.0f}s ===", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--protocol", choices=list(PROTOCOLS), action="append")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    protocols = args.protocol or list(PROTOCOLS)
    models = MODELS if args.all else ([args.model] if args.model else [])
    if not models:
        ap.error("informe --model ou --all")

    for m in models:
        run(m, protocols)


if __name__ == "__main__":
    main()
