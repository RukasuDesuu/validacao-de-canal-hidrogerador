"""A visibilidade do defeito depende da unidade de medida.

Quando um canal é constante no ajuste, ``StandardScaler`` — e o ``OCScaler``
deste trabalho — substituem σ por 1,0. A substituição é apresentada como guarda
numérica contra divisão por zero, mas ela **não é neutra**: o divisor deixa de
carregar a unidade do canal, e o desvio padronizado passa a ser numericamente
igual ao degrau em unidades de engenharia.

A consequência é que o mesmo defeito físico é visível ou invisível conforme o
sensor reporte ampères ou miliampères. Este script demonstra isso e verifica que
o comportamento vem do próprio scikit-learn, não da implementação local.

Uso::

    python -m experiments.injection.units_demo
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.injection.harness import inject, make_split, standardize
from experiments.injection.run import load_skab

VICTIM = "Current"
UNITS = [("kA", 1e-3), ("A", 1.0), ("mA", 1e3), ("µA", 1e6)]


def sklearn_guard() -> None:
    """O mesmo mecanismo, em quatro linhas de scikit-learn puro."""
    X = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])  # canal 0 constante
    sc = StandardScaler().fit(X)
    z = sc.transform([[1.8, 7.0]])[0, 0]
    print("scikit-learn, canal constante no ajuste:")
    print(f"  scale_          = {sc.scale_}")
    print(f"  degrau de 0,80  -> z = {z:.4f}   (isto é: z == degrau em unidades brutas)\n")


def main() -> None:
    sklearn_guard()

    X0 = load_skab()
    train, _, _ = make_split(X0)
    onset = int(len(X0) * 0.75)
    sigma = float(X0.iloc[train][VICTIM].std(ddof=1))
    delta_phys = sigma  # um único defeito físico, o mesmo em todos os casos

    print(f"SKAB, canal '{VICTIM}' travado no ajuste, degrau físico de {delta_phys:.4f} A")
    print(f"{'unidade':>9} {'fator':>8} {'z do degrau':>13} {'contribuição':>14} {'posição':>9}")
    for name, k in UNITS:
        X = X0.copy()
        X[VICTIM] = X[VICTIM] * k
        Xi = inject(X, VICTIM, "latched", delta_phys * k, onset)
        Z, _ = standardize(Xi, train)
        j = list(X.columns).index(VICTIM)
        post = np.arange(len(X)) >= onset
        contrib = (Z[post] ** 2).mean(axis=0)
        rank = int(np.where(np.argsort(-contrib) == j)[0][0]) + 1
        print(f"{name:>9} {k:>8.0e} {delta_phys * k:>13.4g} {contrib[j]:>14.4g} {rank:>7}/{X.shape[1]}")

    print("\nO sensor, o defeito e o modelo são os mesmos nas quatro linhas.")
    print("Só o rótulo da unidade muda — e com ele, a posição do canal culpado.")


if __name__ == "__main__":
    main()
