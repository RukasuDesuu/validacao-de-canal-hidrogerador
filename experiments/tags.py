"""Normalização de prefixo de tag PI e rótulos legíveis de sensor.

As colunas do CSV vêm prefixadas com ``DPP_TRATADA_``; a planilha
``LI-EQUIPAMENTO-MF_SINTOMA__atual.xlsx`` guarda as tags sem prefixo. O
``export_figs.py`` original comparava os dois formatos diretamente, obtinha zero
correspondências e engolia a falha num ``try/except`` mudo — por isso a figura de
SHAP saía rotulada com IDs numéricos crus.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from experiments.config import OC_TAGS, TAG_PREFIX, XLSX_PATH

# Tags operacionais (potência e queda) não recebem prefixo no CSV.
SKIP_PREFIX = re.compile("^(" + "|".join(map(re.escape, OC_TAGS)) + ")$")


def apply_tag_prefix(tag: str, prefix: str = TAG_PREFIX) -> str:
    """Prefixa uma tag da planilha para o formato usado no CSV."""
    if SKIP_PREFIX.match(tag) or tag.startswith(prefix):
        return tag
    return f"{prefix}{tag}"


def strip_tag_prefix(tag: str, prefix: str = TAG_PREFIX) -> str:
    """Remove o prefixo do CSV, devolvendo o formato da planilha."""
    return tag[len(prefix) :] if tag.startswith(prefix) else tag


def short_tag(tag: str) -> str:
    """Identificador curto de uma tag, sem o prefixo do CSV."""
    base = strip_tag_prefix(tag).replace(".Value", "")
    parts = base.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else base


def build_tag_labels(
    xlsx_path: Path = XLSX_PATH,
    columns: list[str] | None = None,
    min_match_rate: float = 0.9,
) -> dict[str, str]:
    """Mapeia tag prefixada -> ``"ABREV — descrição (id curto)"``.

    Args:
        xlsx_path: planilha de sintomas/modos de falha.
        columns: colunas do CSV a rotular. Se informado, valida a taxa de match.
        min_match_rate: fração mínima de ``columns`` que precisa ser resolvida.

    Raises:
        ValueError: se a taxa de correspondência ficar abaixo de ``min_match_rate``.
            Falhar alto aqui é deliberado: o modo de falha silencioso anterior
            produziu uma figura publicada com rótulos errados.
    """
    df = pd.read_excel(xlsx_path, sheet_name="Templates")
    df = df.dropna(subset=["Tags PI System"])

    labels: dict[str, str] = {}
    for _, row in df.iterrows():
        tag = apply_tag_prefix(str(row["Tags PI System"]).strip())
        if tag in labels:
            continue
        abbrev = str(row.get("Abrev. Sintoma", "")).strip()
        name = str(row.get("Sintomas/MF", "")).strip()
        if not name or name == "nan":
            continue
        prefix = f"{abbrev} — " if abbrev and abbrev not in {"-", "nan"} else ""
        labels[tag] = f"{prefix}{name} ({short_tag(tag)})"

    if columns is not None:
        sensor_cols = [c for c in columns if not SKIP_PREFIX.match(c)]
        matched = sum(1 for c in sensor_cols if c in labels)
        rate = matched / len(sensor_cols) if sensor_cols else 0.0
        if rate < min_match_rate:
            raise ValueError(
                f"Apenas {matched}/{len(sensor_cols)} tags do CSV foram resolvidas "
                f"na planilha ({rate:.1%} < {min_match_rate:.0%}). Verifique o "
                f"prefixo '{TAG_PREFIX}' e a aba 'Templates'."
            )
        # Fallback só para o resto, depois da validação.
        for c in sensor_cols:
            labels.setdefault(c, short_tag(c))

    return labels
