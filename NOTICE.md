# Avisos de autoria e licença

Este repositório reúne material de três origens, com termos diferentes.

## 1. Código próprio — MIT

`experiments/`, `mae_model.py` e `data_public/baixar.py`, de Lucas Camargo Ekroth (IFSP, campus
São José dos Campos). Licença MIT, ver `LICENSE`.

## 2. Código da Genesis Data Culture — `fault_diagnosis/`

Os arquivos abaixo são cópias **sem alteração de lógica** do pacote `fault_diagnosis` da
plataforma de monitoramento da Genesis Data Culture, usados aqui para reproduzir os detectores e a rede
bayesiana de produção que servem de comparação:

- `fault_diagnosis/technique/base_detector.py`
- `fault_diagnosis/technique/autoencoder.py` (`MWAutoEncoder`, AutoEncoder por *cluster*)
- `fault_diagnosis/technique/rmwpcasc.py` (`RMWPCA_SC`, PCA robusto de janela móvel)
- `fault_diagnosis/technique/numerical_bayesian_network.py` (rede bayesiana de diagnóstico)
- `fault_diagnosis/training/operational_condition_orchestrator.py` (condições operacionais)
- `fault_diagnosis/data/operational_conditions.json` (limiares de potência e queda das 12 condições da UG-A)

Os três `__init__.py` foram reescritos para carregar apenas esses módulos, e os identificadores
da usina (nomes de *tag* e chaves) foram anonimizados como nos dados (seção 3).

Direitos reservados à Genesis Data Culture. Incluídos aqui apenas para reproduzir os
resultados da monografia e do artigo.

## 3. Dados da usina (UG-A / UHE São Manoel)

- `dados_pi_extraidos.csv` — série horária de uma unidade geradora da UHE São Manoel, aqui
  designada UG-A
  (13.141 registros, 136 *tags* de processo mais potência ativa e queda bruta), extraída da
  camada tratada do historiador de processo (PI System) em junho de 2026.
- `LI-EQUIPAMENTO-MF_SINTOMA__atual.xlsx` — matriz de equipamentos, modos de falha e sintomas
  (FMEA) da unidade.
- `cache/` — escores, atribuições e métricas derivados desses dados pelo pipeline.

Incluídos apenas para reproduzir os resultados da monografia e do artigo.

Os identificadores da usina e os códigos internos foram anonimizados: a chave da unidade vira
`UGA`; as *tags* numéricas do historiador viram `UGA_T001` a `UGA_T100`, na ordem das colunas
do CSV; as de entreferro e vibração mantêm só a descrição (`UGA_GERADOR_…`); as de potência
ativa e queda bruta viram `UGA_POTENCIA_ATIVA_MW` e `UGA_QUEDA_BRUTA_M`; e os seis
equipamentos viram `EQ01` a `EQ06`. As siglas dos *clusters* (CFA, TARS etc.) e todos os
valores são os originais. A planilha traz só a aba, as colunas e os equipamentos que o
pipeline lê.

A camada tratada do historiador foi reprocessada pela operadora depois da extração, sem cópia de
segurança; as seis *tags* elétricas do episódio de maio de 2026 não têm mais dados nela. Este
CSV é, portanto, um instantâneo de junho de 2026, e não pode ser regenerado a partir do PI.

## 4. Conjuntos públicos (não redistribuídos)

`data_public/baixar.py` baixa os arquivos dos repositórios originais, num commit fixo, e confere o
SHA-256:

- **SKAB** — Skoltech Anomaly Benchmark, Katser e Kozitsin (2020), <https://github.com/waico/SKAB>,
  licença GPL-3.0.
- **SMD** — Server Machine Dataset, Su et al. (2019), <https://github.com/NetManAIOps/OmniAnomaly>,
  licença MIT.
