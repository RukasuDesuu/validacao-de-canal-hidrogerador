"""Baixa os dois conjuntos públicos usados na injeção controlada (Seção 4.5 da monografia).

Os arquivos não são redistribuídos aqui: vêm dos repositórios originais, num commit fixo,
e são conferidos por SHA-256.

    python data_public/baixar.py
"""

import hashlib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [
    (  # SKAB, Skoltech Anomaly Benchmark (GPL-3.0), Katser e Kozitsin (2020)
        "https://raw.githubusercontent.com/waico/SKAB/1801b97459937a872c789add44e64511b586b1f2/data/anomaly-free/anomaly-free.csv",
        HERE / "skab" / "anomaly-free.csv",
        "7641a8742957100ef424dd97adfb3eeab27d4d180645fe2b3c49b7f7146ae9a6",
    ),
    (  # SMD, Server Machine Dataset (MIT), Su et al. (2019), série machine-1-1 de treino
        "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/3da9a2e9792adadef58e292ada348d8dc5ae506d/ServerMachineDataset/train/machine-1-1.txt",
        HERE / "smd" / "machine-1-1-train.txt",
        "ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753",
    ),
]


def main() -> None:
    for url, dest, sha in FILES:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"SHA-256 divergente em {dest}: {got}")
        print(f"ok {dest.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
