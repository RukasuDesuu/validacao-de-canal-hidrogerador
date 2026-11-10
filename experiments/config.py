"""Constantes centrais dos experimentos do artigo.

Todo número que aparece no artigo deve vir daqui ou de `metrics.parquet` — nunca
hardcoded no texto ou nos scripts de figura.
"""

import os
from pathlib import Path

# --- Caminhos -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dados_pi_extraidos.csv"
XLSX_PATH = ROOT / "LI-EQUIPAMENTO-MF_SINTOMA__atual.xlsx"
CACHE_DIR = ROOT / "cache"


def _output_dir(env_var: str, fallback: str) -> Path:
    """Pasta de saída de tabelas e figuras: a variável de ambiente, senão ``output/``."""
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    out = ROOT / "output" / fallback
    out.mkdir(parents=True, exist_ok=True)
    return out


# Tabelas e figuras do artigo e da monografia.
ARTICLE_DIR = _output_dir("TCC_ARTICLE_DIR", "artigo")
MONOGRAFIA_DIR = _output_dir("TCC_MONOGRAFIA_DIR", "monografia")

# --- Colunas e tags -----------------------------------------------------------
TIME_COL = "Data - Hora"
TAG_PREFIX = "DPP_TRATADA_"
POWER_TAG = "UGA_POTENCIA_ATIVA_MW"
HEAD_TAG = "UGA_QUEDA_BRUTA_M"
OC_TAGS = [POWER_TAG, HEAD_TAG]
OC_KEY = "UGA"

# Corrente de fase A, o canal que ilustra o episódio E2 (linha do tempo da monografia).
CFA_TAG = "DPP_TRATADA_UGA_T070"

EQUIPMENT_IDS = [
    "EQ01",
    "EQ02",
    "EQ03",
    "EQ04",
    "EQ05",
    "EQ06",
]

# --- Pré-processamento --------------------------------------------------------
MIN_POWER_MW = 50.0
SAMPLING = "1h"

# Modelos globais: janela de 24 h com passo 1 h. O passo unitário é necessário
# porque E2 tem duração de horas — um passo de 6 h daria resolução grossa demais.
WINDOW = 24
STRIDE = 1

# As janelas de treino são subamostradas: com passo 1 h a sobreposição entre
# janelas vizinhas é de 95,8%, o que enche o conjunto de quase-duplicatas sem
# acrescentar informação. A pontuação continua em passo de 1 h.
TRAIN_STRIDE = 3

# Fração final do bloco de treino reservada para calibrar o limiar. O modelo
# nunca a vê durante o ajuste: seu erro de reconstrução no bloco de ajuste é
# sistematicamente menor que em qualquer dado novo, e um limiar estimado ali
# seria excedido quase sempre fora da amostra. Reservar o trecho final — e não
# um trecho aleatório — dá ao limiar as estatísticas saudáveis mais recentes.
CALIBRATION_FRAC = 0.25

# Baselines por cluster mantêm os hiperparâmetros de produção.
AE_CLUSTER_WINDOW = 361
PCA_CLUSTER_WINDOW = 180
AE_CLUSTER_LATENT = 3
AE_CLUSTER_HIDDEN = (64, 32)
AE_CLUSTER_EPOCHS = 50
NUM_STDS = 3.0

# --- Treino dos modelos globais ----------------------------------------------
MAE_D_MODEL = 64
MAE_LAYERS = 2
MAE_NHEAD = 4
MAE_EPOCHS = 20
MAE_BATCH = 64
MAE_LR = 1e-3
MASK_RATIO = 0.5
ABLATION_MASK_RATIOS = [0.0, 0.25, 0.5, 0.75]
SEEDS = [42, 43, 44]

# --- Protocolos ---------------------------------------------------------------
# P1: bloco de saúde curado. Treino/scalers/UCL só em H1; pontuação da série inteira.
#     É o único protocolo capaz de avaliar E1 (não existe bloco de saúde por OC
#     anterior a fev/2025 — 0 OCs atingem min_training_points até 15/01/2025).
# P2: holdout estritamente causal. Nenhuma informação futura; avalia apenas E2.
PROTOCOLS = {
    "P1": {
        "train": ("2025-03-01 00:00", "2025-12-31 23:00"),
        "score": None,  # série inteira
        "evaluates": ["E1", "E2"],
        "label": "P1 (bloco de saúde curado)",
    },
    "P2": {
        "train": ("2024-12-07 03:00", "2026-01-31 23:00"),
        "score": ("2026-02-01 00:00", "2026-06-07 21:00"),
        "evaluates": ["E2"],
        "label": "P2 (holdout causal)",
    },
}

# --- Persistência de alarme ---------------------------------------------------
PERSIST_K = 3  # k de n horas acima do limiar
PERSIST_N = 5

# --- Bootstrap ----------------------------------------------------------------
BOOTSTRAP_BLOCK_H = 168
BOOTSTRAP_N = 1000

# --- Valores esperados (asserts de sanidade) ----------------------------------
EXPECTED_N_RAW = 13141
EXPECTED_N_FILTERED = 10933
EXPECTED_N_CLUSTERS = 33
EXPECTED_N_FAULT_MODES = 13
EXPECTED_N_TAGS = 136
