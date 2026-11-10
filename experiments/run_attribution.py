"""Atribuição do erro do evento E2 nos detectores além do MAE.

O SHAP (``run_explain.py``) só cobre o MAE. Esta etapa mede, para os demais detectores,
onde o erro/alarme está, com a ferramenta adequada a cada família:

* ``--part global``: modelos globais que reconstroem a janela (MAE, AE-Transformer sem
  máscara, AE denso). O escore é a média do erro quadrático sobre tempo e sensores, logo a
  decomposição exata por *tag* (erro quadrático médio de cada tag) é uma atribuição sem
  aproximação. Grava ``cache/attrib_global__{modelo}__{evento}.parquet``.
* ``--part cluster``: modelos de produção, que pontuam por cluster x condição e agregam
  pelo máximo entre clusters. Aqui a pergunta natural é qual cluster carrega o alarme.
  Grava a razão Q/UCL de cada cluster, hora a hora:
  ``cache/attrib_cluster__{modelo}__{protocolo}.parquet``.

* ``--part classic``: Isolation Forest e One-Class SVM, que operam sobre componentes principais
  da janela achatada e não têm decomposição direta por tag. Atribui-se o escore que o
  detector de fato usa (``-decision_function``) por valores de Shapley com cada tag (seus 24
  instantes juntos) como jogador, estimados por amostragem de permutações contra janelas
  saudáveis de fundo. Modelo-agnóstico; a soma das contribuições fecha exatamente a diferença
  entre o escore da janela e o do fundo. Grava ``cache/attrib_global__{iforest,ocsvm}__{evento}.parquet``
  (mesmo formato dos modelos globais). Só a monografia usa esta parte.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.clusters import load_clusters_from_xlsx
from experiments.config import CACHE_DIR, MAE_BATCH, MAE_EPOCHS, MAE_LR, MASK_RATIO, SEEDS
from experiments.data import prepare
from experiments.ground_truth import EVENTS_BY_ID
from experiments.models.common import build_windows

BATCH = 256


def _event_rows(ws, event_id: str) -> np.ndarray:
    ev = EVENTS_BY_ID[event_id]
    sel = (ws.index >= pd.Timestamp(ev.start)) & (ws.index <= pd.Timestamp(ev.end))
    return np.flatnonzero(np.asarray(sel))


def _per_tag_error(recon_fn, X: np.ndarray) -> np.ndarray:
    """Erro quadrático médio no tempo, por janela e por tag: (n_janelas, n_tags)."""
    out = np.empty((len(X), X.shape[2]), dtype=float)
    with torch.no_grad():
        for s in range(0, len(X), BATCH):
            xb = torch.tensor(X[s : s + BATCH], dtype=torch.float32)
            rb = recon_fn(xb)
            out[s : s + len(xb)] = ((xb - rb) ** 2).mean(dim=1).numpy()
    return out


def _summarize(err: np.ndarray, tags: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(err, columns=tags)
    out = pd.DataFrame({"median": df.median(), "q1": df.quantile(0.25), "q3": df.quantile(0.75), "mean": df.mean()})
    out["share"] = out["median"] / out["median"].sum()
    out["n_windows"] = len(df)
    return out.sort_values("median", ascending=False)


def run_global(protocol: str, events: list[str]) -> None:
    from experiments.models.ae_dense import DenseAE
    from experiments.models.mae_global import torch_model_for_shap

    ds = prepare()
    seed = SEEDS[0]
    jobs = {}

    for name, mask in (("mae_m050", MASK_RATIO), ("ae_transformer_sem_mascara", 0.0)):
        t0 = time.time()
        print(f"treinando {name}...", flush=True)
        model, ws = torch_model_for_shap(ds, protocol, mask_ratio=mask, seed=seed)
        model.eval()
        jobs[name] = (lambda xb, m=model: m(xb)[0], ws)
        print(f"  {time.time() - t0:.0f}s", flush=True)

    print("treinando ae_denso...", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ws = build_windows(ds, protocol)
    flat = ws.flat()
    dense = DenseAE(flat.shape[1])
    opt = torch.optim.Adam(dense.parameters(), lr=MAE_LR)
    loader = DataLoader(TensorDataset(torch.tensor(flat[ws.fit_mask], dtype=torch.float32)), batch_size=MAE_BATCH, shuffle=True)
    dense.train()
    for _ in range(MAE_EPOCHS):
        for (b,) in loader:
            opt.zero_grad()
            torch.mean((dense(b) - b) ** 2).backward()
            opt.step()
    dense.eval()
    shape = ws.X.shape[1:]
    jobs["ae_denso"] = (lambda xb, m=dense, s=shape: m(xb.reshape(len(xb), -1)).reshape(len(xb), *s), ws)

    for name, (recon_fn, ws_) in jobs.items():
        tags = list(ws_.scaler.columns_)
        for ev in events:
            rows = _event_rows(ws_, ev)
            if len(rows) == 0:
                continue
            summary = _summarize(_per_tag_error(recon_fn, ws_.X[rows]), tags)
            path = CACHE_DIR / f"attrib_global__{name}__{ev}.parquet"
            summary.to_parquet(path)
            print(f"  {name} {ev}: {len(rows)} janelas; topo {list(summary.index[:2])} -> {path.name}", flush=True)


def run_cluster(protocol: str, models: list[str]) -> None:
    from experiments.models import ae_cluster, pca_cluster
    from experiments.models.cluster_common import cluster_frames

    ds = prepare()
    clusters, _ = load_clusters_from_xlsx()
    impl = {
        "rmwpca": (pca_cluster._fit_one, pca_cluster.PCA_CLUSTER_WINDOW, {}),
        "ae_cluster": (ae_cluster._fit_one, ae_cluster.AE_CLUSTER_WINDOW, {"seed": SEEDS[0]}),
    }
    for name in models:
        fit_one, window, kw = impl[name]
        t0 = time.time()
        per_cluster: dict[str, pd.Series] = {}
        for n, cluster in enumerate(clusters, 1):
            parts = []
            for _label, health, score in cluster_frames(ds, cluster, protocol, window):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        idx, q_ratio, _ = fit_one(health, score, window, **kw) if kw else fit_one(health, score, window)
                except Exception:  # noqa: BLE001 — condição degenerada não invalida o cluster
                    continue
                parts.append(pd.Series(q_ratio, index=idx))
            if parts:
                per_cluster[cluster["abbrev"]] = pd.concat(parts).sort_index()
            print(f"  [{name} {n}/{len(clusters)}] {cluster['abbrev']} {time.time() - t0:.0f}s", flush=True)
        frame = pd.DataFrame(per_cluster).reindex(ds.index)
        path = CACHE_DIR / f"attrib_cluster__{name}__{protocol}.parquet"
        frame.to_parquet(path)
        print(f"{name}: {frame.shape} -> {path.name}", flush=True)


def _shapley_by_tag(score_fn, X_events: np.ndarray, backgrounds: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """Shapley por tag (jogador = tag inteira) por amostragem de permutações.

    Para cada janela ``x`` e cada permutação, parte de uma janela de fundo ``b`` e vai
    trocando tag a tag os valores de ``b`` pelos de ``x``; a variação do escore a cada troca
    é a contribuição marginal. Usa permutações antitéticas (a ordem e sua inversa).
    Retorna ``(n_janelas, n_tags)``; a soma de cada linha é ``f(x) - média de f(b)`` sobre
    os fundos sorteados.
    """
    n, _, n_tags = X_events.shape
    phi = np.zeros((n, n_tags))
    for i, x in enumerate(X_events):
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(n_tags)
            b = backgrounds[rng.integers(len(backgrounds))]
            for order in (perm, perm[::-1]):
                masks = np.zeros((n_tags + 1, n_tags), dtype=bool)
                for k in range(1, n_tags + 1):
                    masks[k:, order[k - 1]] = True
                hybrid = np.where(masks[:, None, :], x[None], b[None])
                sc = score_fn(hybrid)
                phi[i, order] += np.diff(sc)
                count += 1
        phi[i] /= count
    return phi


def run_classic(protocol: str, events: list[str], n_perm: int = 24, n_bg: int = 8, max_windows: int = 48) -> None:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    from experiments.cache import load_scores
    from experiments.models.classic import CONTAMINATION, NU, PCA_VARIANCE

    ds = prepare()
    seed = SEEDS[0]
    ws = build_windows(ds, protocol)
    flat = ws.flat()
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=seed).fit(flat[ws.fit_mask])
    train = pca.transform(flat[ws.fit_mask])
    iforest = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=seed).fit(train)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=NU).fit(train)
    shape = ws.X.shape[1:]
    models = {
        "iforest": lambda w: -iforest.decision_function(pca.transform(w.reshape(len(w), -1))),
        "ocsvm": lambda w: -ocsvm.decision_function(pca.transform(w.reshape(len(w), -1))),
    }
    tags = list(ws.scaler.columns_)

    rng = np.random.default_rng(seed)
    fit_idx = np.flatnonzero(ws.fit_mask)
    backgrounds = ws.X[rng.choice(fit_idx, size=n_bg, replace=False)]

    for name, fn in models.items():
        cached = load_scores(name, protocol)
        probe = np.flatnonzero(np.isin(ws.index, cached.index[:200]))[:50]
        ref = pd.Series(cached.score, index=cached.index).reindex(ws.index[probe]).to_numpy()
        diff = float(np.nanmax(np.abs(fn(ws.X[probe]) - ref)))
        print(f"{name}: refit reproduz o escore do cache (diferença máx. {diff:.2e})", flush=True)
        for ev in events:
            rows = _event_rows(ws, ev)
            if len(rows) == 0:
                continue
            if len(rows) > max_windows:
                rows = rows[np.linspace(0, len(rows) - 1, max_windows).astype(int)]
            t0 = time.time()
            phi = _shapley_by_tag(fn, ws.X[rows], backgrounds, n_perm, rng)
            resid = float(np.max(np.abs(phi.sum(axis=1) - (fn(ws.X[rows]) - np.mean([fn(b[None])[0] for b in backgrounds])))))
            summary = _summarize(np.abs(phi), tags)
            path = CACHE_DIR / f"attrib_global__{name}__{ev}.parquet"
            summary.to_parquet(path)
            print(f"  {name} {ev}: {len(rows)} janelas, {time.time() - t0:.0f}s, "
                  f"desvio de eficiência máx. {resid:.2f}; topo {list(summary.index[:2])} -> {path.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["global", "cluster", "classic"], required=True)
    ap.add_argument("--protocol", default="P1")
    ap.add_argument("--events", nargs="+", default=["E1", "E2"])
    ap.add_argument("--models", nargs="+", default=["rmwpca", "ae_cluster"])
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.part == "global":
        run_global(args.protocol, args.events)
    elif args.part == "classic":
        run_classic(args.protocol, args.events)
    else:
        run_cluster(args.protocol, args.models)


if __name__ == "__main__":
    main()
