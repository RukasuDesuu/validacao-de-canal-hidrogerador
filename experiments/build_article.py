"""Regenera todas as figuras e tabelas do artigo a partir do cache.

Não treina nada — se um modelo estiver ausente do cache, a figura correspondente
é pulada com aviso. Para recomputar, use ``python -m experiments.run_all``.
"""

from __future__ import annotations

import traceback

from experiments import tables
from experiments.figures import fig_ablation, fig_case_cfa, fig_dataset, fig_scores_panel, fig_shap

def _verify() -> str:
    from experiments.verify_claims import main as verify

    if verify() != 0:
        raise SystemExit("há afirmações do artigo que não conferem com os dados")
    return "todas as afirmações conferem"


STEPS = [
    ("tabela do conjunto de dados", lambda: tables.dataset_table()),
    ("tabela de métricas", lambda: tables.metrics_table()),
    ("métricas consolidadas", lambda: tables.dump_metrics()),
    ("figura do conjunto de dados", lambda: fig_dataset.build()[0]),
    ("painel de scores (P1)", lambda: fig_scores_panel.build("P1")),
    ("painel de scores (P2)", lambda: fig_scores_panel.build("P2")),
    ("estudo de caso", lambda: fig_case_cfa.build()),
    ("SHAP", lambda: fig_shap.build()),
    ("ablação", lambda: fig_ablation.build()),
    ("conferência dos números do artigo", _verify),
]


def main() -> None:
    failures = []
    for name, fn in STEPS:
        try:
            print(f"[ok]    {name}: {fn()}", flush=True)
        except SystemExit as exc:
            print(f"[pulado] {name}: {exc}", flush=True)
            failures.append(name)
        except Exception:
            print(f"[ERRO]  {name}", flush=True)
            traceback.print_exc()
            failures.append(name)
    if failures:
        print(f"\npendentes: {', '.join(failures)}")


if __name__ == "__main__":
    main()
