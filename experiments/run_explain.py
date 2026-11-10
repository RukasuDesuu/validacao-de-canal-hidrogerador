"""Calcula e cacheia as atribuições SHAP dos dois eventos.

Etapa separada porque exige treinar um MAE próprio; o resultado vai para
``cache/shap__{evento}.parquet`` e as figuras nunca reexecutam o explicador.
"""

from __future__ import annotations

import argparse

from experiments.clusters import load_clusters_from_xlsx
from experiments.config import CACHE_DIR, MASK_RATIO, SEEDS
from experiments.data import prepare
from experiments.explain import shap_by_cluster, shap_for_event
from experiments.models.mae_global import torch_model_for_shap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="P1")
    ap.add_argument("--events", nargs="+", default=["E1", "E2"])
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = prepare()
    clusters, _ = load_clusters_from_xlsx()

    print("treinando o MAE para explicabilidade...", flush=True)
    model_ws = torch_model_for_shap(ds, args.protocol, mask_ratio=MASK_RATIO, seed=SEEDS[0])

    for event_id in args.events:
        print(f"SHAP para {event_id}...", flush=True)
        per_tag = shap_for_event(ds, event_id, protocol=args.protocol, model_ws=model_ws)
        per_tag.to_parquet(CACHE_DIR / f"shap__{event_id}.parquet")
        shap_by_cluster(per_tag, clusters).to_parquet(CACHE_DIR / f"shap_cluster__{event_id}.parquet")
        print(f"  {per_tag['n_windows'].iloc[0]} janelas; topo: {list(per_tag.index[:3])}", flush=True)


if __name__ == "__main__":
    main()
