# Validação de canal antes da normalização em hidrogeradores

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24147471.svg)](https://doi.org/10.5281/zenodo.24147471)

Código e dados que reproduzem os resultados da monografia *Detecção e Diagnóstico de Falhas em Hidrogeradores por Aprendizado Não Supervisionado e Explicabilidade* (Lucas Camargo Ekroth,
IFSP, campus São José dos Campos, 2026) e do artigo derivado dela.

**Resumo.** Seis sinais elétricos de um hidrogerador ficaram constantes por 17 meses e depois
mudaram de valor. Os detectores de anomalias normalizam cada sensor pelo desvio padrão em operação
saudável, e um canal constante tem desvio padrão nulo. Por isso nenhum dos sete detectores avaliados
atribuiu o erro aos canais que mudaram. Um teste que conta valores distintos antes da normalização
detecta o episódio, e o mecanismo se reproduz por injeção controlada em dois conjuntos públicos.

*In English:* code and data for a case study showing that zero-variance channels are invisible to
standard-score-normalized anomaly detectors in a hydrogenerator, and that a scale-invariant
cardinality/novelty check applied before normalization detects the event.

## Instalação

Python 3.12, CPU (sem GPU).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu   # opcional: evita baixar a versão CUDA
pip install -r requirements.txt
python data_public/baixar.py        # SKAB e SMD, dos repositórios originais (ver NOTICE.md)
```

## Comandos

Todos rodam a partir da raiz do repositório. A fronteira é rígida: os modelos gravam escores em
`cache/`, e as tabelas e figuras só leem de lá. O `cache/` publicado já contém os resultados
finais, então os quatro últimos comandos funcionam sem re-treinar nada.

```bash
python -m experiments.run_all --all                      # treina e pontua os modelos (algumas horas em CPU)
python -m experiments.run_explain                        # atribuições SHAP dos dois eventos
python -m experiments.run_attribution --part global      # atribuição nos
python -m experiments.run_attribution --part cluster     #   demais
python -m experiments.run_attribution --part classic     #   detectores
python -m experiments.injection.run --dataset skab       # injeção controlada nos
python -m experiments.injection.run --dataset smd        #   conjuntos públicos
python -m experiments.attribution_table --com-classicos  # tabela de atribuição
python -m experiments.run_diagnosis                      # diagnóstico por rede bayesiana
python -m experiments.build_article                      # tabelas e figuras compartilhadas com o artigo
python -m experiments.figures.mono_figs                  # figuras exclusivas da monografia
python -m experiments.verify_claims                      # confere os números do texto contra os dados
```

Para rodar um modelo isolado: `python -m experiments.run_all --model sensor_check --protocol P2`
(modelos em `experiments/run_all.py`, `MODELS`).

As tabelas e figuras vão para `output/artigo/` e `output/monografia/`. As variáveis de ambiente
`TCC_ARTICLE_DIR` e `TCC_MONOGRAFIA_DIR` mudam esse destino.

## Conteúdo

| Caminho | O que é |
|---|---|
| `experiments/` | pipeline completo; toda constante de projeto está em `experiments/config.py` (detalhes em `experiments/README.md`) |
| `mae_model.py` | o *Masked Autoencoder* (`TimeSeriesMAE`) |
| `fault_diagnosis/` | os cinco módulos da plataforma de produção usados como comparação (ver NOTICE.md) |
| `dados_pi_extraidos.csv` | série horária da unidade UG-A da UHE São Manoel, 13.141 registros, 136 *tags* de processo |
| `LI-EQUIPAMENTO-MF_SINTOMA__atual.xlsx` | matriz de modos de falha e sintomas (FMEA) da unidade |
| `cache/` | escores, atribuições e métricas finais |
| `data_public/baixar.py` | download verificado do SKAB e do SMD |

**Proveniência do episódio de maio de 2026 (E2).** A série foi extraída da camada tratada do
historiador em junho de 2026. Essa camada foi depois reprocessada pela operadora, sem cópia de
segurança, e as seis *tags* elétricas do episódio não têm mais dados nela. O CSV deste repositório
é a única cópia da série tal como a plataforma a processava.

## Licença

Código próprio sob MIT (`LICENSE`). Código da Genesis Data Culture e dados da usina sob os
termos em `NOTICE.md`.

## Como citar

EKROTH, Lucas Camargo. **Detecção e diagnóstico de falhas em hidrogeradores por aprendizado
não supervisionado e explicabilidade**. 2026. Trabalho de Conclusão de Curso (Bacharelado em
Engenharia de Controle e Automação) – Instituto Federal de Educação, Ciência e Tecnologia de
São Paulo, São José dos Campos, 2026.

Para citar este repositório (código e dados), use o DOI do arquivo no Zenodo:
[10.5281/zenodo.24147471](https://doi.org/10.5281/zenodo.24147471). Os metadados estão em `CITATION.cff`.

A referência do artigo será acrescentada quando ele for publicado.
