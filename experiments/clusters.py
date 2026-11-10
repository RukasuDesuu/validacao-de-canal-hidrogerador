"""Construção dos clusters de sensores e modos de falha a partir da planilha.

Função reutilizável, com asserts de sanidade.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.config import (
    EQUIPMENT_IDS,
    EXPECTED_N_CLUSTERS,
    EXPECTED_N_FAULT_MODES,
    EXPECTED_N_TAGS,
    XLSX_PATH,
)
from experiments.tags import apply_tag_prefix

# A planilha de origem (``LI-EQUIPAMENTO-MF_SINTOMA__atual.xlsx``) grafa vários nomes de
# sintoma e de modo de falha sem acentuação. A planilha não é alterada (é a fonte de
# verdade dos dados); este dicionário corrige apenas o texto exibido em figuras e tabelas.
_NAME_FIXES = {
    "Carga d'agua": "Carga d'água",
    "Fluxo oleo saida skid MGG": "Fluxo óleo saída skid MGG",
    "Nivel ar/oleo": "Nível ar/óleo",
    "Nivel oleo cuba inferior MGG": "Nível óleo cuba inferior MGG",
    "Nivel Reservatorio": "Nível Reservatório",
    "Posicao Distribuidor": "Posição Distribuidor",
    "Posicao Distribuidor TSLG-N": "Posição Distribuidor TSLG-N",
    "Pressao da Linha": "Pressão da Linha",
    "Pressao oleo injecao ME": "Pressão óleo injeção ME",
    "Posicao das pas": "Posição das pás",
    "Referencia Abertura Efetiva": "Referência Abertura Efetiva",
    "Referencia Abertura Efetiva TSLG-S": "Referência Abertura Efetiva TSLG-S",
    "Temperatura do ar de resfriamento de saida": "Temperatura do ar de resfriamento de saída",
    "Temperatura agua de Saida": "Temperatura água de Saída",
    "Temperatura entrada oleo trocador MGG": "Temperatura entrada óleo trocador MGG",
    "Temperatura oleo cuba MGG": "Temperatura óleo cuba MGG",
    "Temp saida agua trocador MGG": "Temp saída água trocador MGG",
    "Temp saida oleo trocador MGG": "Temp saída óleo trocador MGG",
    "Vazao agua resfriamento MGG": "Vazão água resfriamento MGG",
    "Vibracao": "Vibração",
    "Vibracao MGG": "Vibração MGG",
    "Vazao oleo saida skid ME": "Vazão óleo saída skid ME",
    "Sobreaquecimento (desvios do sistema de ventilacao do gerador)": "Sobreaquecimento (desvios do sistema de ventilação do gerador)",
    "Obstrucao/entupimento": "Obstrução/entupimento",
    "Falha na lubrificacao dos mancais": "Falha na lubrificação dos mancais",
    "Parafusos de fixacao soltos": "Parafusos de fixação soltos",
    "Perda de pressao": "Perda de pressão",
}


def _fix_name(name: str) -> str:
    return _NAME_FIXES.get(name, name)


def load_clusters_from_xlsx(
    xlsx_path: Path = XLSX_PATH,
    equipment_ids: list[str] | None = None,
    validate: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Devolve ``(CLUSTERS, FAULT_MODES)``.

    Cada cluster é ``{id, name, abbrev, tags, fault_modes}`` com as tags já
    prefixadas para casar com as colunas do CSV.
    """
    equipment_ids = equipment_ids or EQUIPMENT_IDS
    df = pd.read_excel(xlsx_path, sheet_name="Templates")
    df = df[df["ID Equipamento"].isin(equipment_ids)].copy()

    # Modos de falha: linhas de nível de equipamento (abreviação de sintoma == '-').
    df_fm = df[df["Abrev. Sintoma"] == "-"].drop_duplicates(subset=["ID Equipamento", "Sintomas/MF"])
    fault_modes = [
        {
            "id": i,
            "name": _fix_name(row["Sintomas/MF"]),
            "local": row["Local de Instalação"],
            "equip": row["ID Equipamento"],
        }
        for i, (_, row) in enumerate(df_fm.iterrows(), start=1)
    ]

    equip_to_fm: dict[str, list[int]] = {}
    for fm in fault_modes:
        equip_to_fm.setdefault(fm["equip"], []).append(fm["id"])

    df_cl = df[(df["Abrev. Sintoma"] != "-") & df["Abrev. Sintoma"].notna() & df["Tags PI System"].notna()]

    grouped: dict[str, dict] = {}
    for _, row in df_cl.iterrows():
        abbrev = row["Abrev. Sintoma"]
        tag = row["Tags PI System"]
        entry = grouped.setdefault(abbrev, {"name": _fix_name(row["Sintomas/MF"]), "tags": [], "equip_ids": set()})
        entry["equip_ids"].add(row["ID Equipamento"])
        if tag not in entry["tags"]:
            entry["tags"].append(tag)

    clusters = []
    for i, (abbrev, info) in enumerate(sorted(grouped.items()), start=1):
        fm_ids = sorted({fid for eq in info["equip_ids"] for fid in equip_to_fm.get(eq, [])})
        clusters.append(
            {
                "id": i,
                "name": info["name"],
                "abbrev": abbrev,
                "tags": [apply_tag_prefix(t) for t in info["tags"]],
                "fault_modes": fm_ids,
            }
        )

    if validate:
        n_tags = len({t for c in clusters for t in c["tags"]})
        assert len(clusters) == EXPECTED_N_CLUSTERS, f"{len(clusters)} clusters != {EXPECTED_N_CLUSTERS}"
        assert len(fault_modes) == EXPECTED_N_FAULT_MODES, f"{len(fault_modes)} modos != {EXPECTED_N_FAULT_MODES}"
        assert n_tags == EXPECTED_N_TAGS, f"{n_tags} tags únicas != {EXPECTED_N_TAGS}"

    return clusters, fault_modes


def cluster_by_abbrev(clusters: list[dict]) -> dict[str, dict]:
    """Índice ``abbrev -> cluster``."""
    return {c["abbrev"]: c for c in clusters}


# Clusters da malha de medição elétrica afetados pelo evento E2.
# Atenção: TFABG/TFBCG/TFCAG são TENSÃO de fase (AB/BC/CA), não temperatura de
# mancal guia — a planilha é a fonte de verdade, e os docs antigos erravam nisso.
ELECTRICAL_ABBREVS = ["CFA", "CFB", "CFC", "TFABG", "TFBCG", "TFCAG"]
