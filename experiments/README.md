# `experiments/` — pipeline de resultados do artigo

Há uma fronteira dura: **modelos escrevem escores em `cache/`; figuras e tabelas só leem do cache.**
Regenerar uma figura nunca re-treina nada.

Todos os comandos rodam a partir da raiz do repositório, com o ambiente do `requirements.txt`
(ver o `README.md` principal):

```bash
PY=python
```

## Fluxo completo

```bash
$PY -m experiments.run_all --all            # treina e pontua tudo (horas, em CPU)
$PY -m experiments.run_explain              # atribuições SHAP dos dois eventos
$PY -m experiments.build_article            # tabelas e figuras -> output/artigo/
$PY -m experiments.verify_claims            # confere os números do main.tex
```

Para rodar um modelo isolado: `--model {mae, mae_ablation, ae_dense, classic, pca_cluster,
ae_cluster, sigma_floor}`, opcionalmente com `--protocol {P1,P2}`.

## Onde ficam as decisões

Toda constante está em `config.py` — não há número mágico espalhado pelos módulos. Os pontos
que mais afetam os resultados:

| Constante | Papel |
|---|---|
| `MIN_POWER_MW` | descarta máquina parada e baixa carga, que dominariam os escores |
| `WINDOW` / `STRIDE` | janela de 24 h com passo de 1 h para os modelos globais |
| `TRAIN_STRIDE` | subamostra as janelas de ajuste (com passo 1 h a sobreposição é de 95,8%) |
| `CALIBRATION_FRAC` | quarto final do bloco de saúde, reservado para calibrar o limiar |
| `PROTOCOLS` | blocos de treino e pontuação de P1 e P2 |

## Três armadilhas que este pipeline evita

**Limiar estimado dentro da amostra.** O erro de reconstrução sobre os próprios dados de ajuste
é sistematicamente menor que sobre dados novos. Um limite `µ + 3σ` calculado ali é excedido em
mais de 80% das horas saudáveis fora da amostra — para todos os modelos, inclusive os que não
são redes neurais. Daí a divisão ajuste/calibração em `models/common.py`.

**Deriva do processo.** Mesmo com calibração fora da amostra, o escore saudável cresce ao longo
de meses. `metrics.rolling_ucl` reestima o limite sobre uma janela de retaguarda estritamente
passada; use `metrics.adaptive_flags` para obter razão, limiar e alarmes de uma vez, e não
recalcule isso em cada figura.

**Casamento de tags.** As colunas do CSV têm o prefixo `DPP_TRATADA_`; a planilha não. A versão
anterior comparava os dois formatos direto, obtinha zero correspondências e engolia a falha num
`try/except` — a figura de SHAP saiu publicada com identificadores numéricos crus.
`tags.build_tag_labels` levanta `ValueError` se a taxa de casamento cair abaixo de 90%.

## Diferenças em relação ao pipeline de produção

Registradas porque afetam a comparabilidade dos números:

- Os limites de controle saem do bloco de saúde, não da série inteira. No sistema de produção, o
  `health_data` é calculado e descartado, e o `fit` recebe tudo — os limites acabam
  incorporando os eventos que deveriam detectar.
- Os modelos por cluster (`pca_cluster`, `ae_cluster`) são a exceção deliberada: mantêm o
  comportamento de produção, para representarem o baseline real.
- As janelas só são emitidas dentro de trechos de amostragem contígua; o filtro de potência
  abre 30 segmentos, e o maior buraco tem 20 dias.
