# Previsão de Churn — Rede Neural com Pipeline Profissional End-to-End

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)]()
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)]()
[![Lint](https://img.shields.io/badge/ruff-passing-brightgreen)]()

Pipeline de Machine Learning completo, do dado bruto ao modelo servido via
API, para prever a probabilidade de cancelamento (*churn*) de clientes de
uma operadora de telecomunicações. Projeto desenvolvido para o **Tech
Challenge — Fase 1**.

> **Modelo central:** rede neural MLP (PyTorch), comparada com baselines
> (Scikit-Learn) e com todos os experimentos rastreados no **MLflow**.

---

## Sumário

- [Contexto de negócio](#contexto-de-negócio)
- [Resultados principais](#resultados-principais)
- [Arquitetura do projeto](#arquitetura-do-projeto)
- [Setup e instalação](#setup-e-instalação)
- [Como usar](#como-usar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Testes e qualidade](#testes-e-qualidade)
- [Documentação adicional](#documentação-adicional)

---

## Contexto de negócio

Uma operadora de telecomunicações está perdendo clientes em ritmo
acelerado. Este projeto constrói um modelo preditivo que classifica
clientes por risco de cancelamento, permitindo que o time de retenção
priorize ações proativas (descontos, contato direto) **antes** que o
cliente cancele — em vez de reagir depois do fato.

- **Dataset:** [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.043 clientes, 21 colunas.
- **Métrica técnica primária:** ROC-AUC (com PR-AUC, F1, Precision e Recall como apoio).
- **Métrica de negócio:** custo esperado = `FP × R$30 + FN × R$800` (ver [Model Card](docs/model_card.md), seção 5, para as premissas completas).

## Resultados principais

| Modelo | ROC-AUC | F1 | Recall | Custo de negócio (R$) |
|---|---|---|---|---|
| Dummy (baseline trivial) | 0.500 | 0.000 | 0.000 | 299.200 |
| Regressão Logística | 0.841 | 0.614 | 0.783 | 73.440 |
| Random Forest | 0.842 | 0.624 | 0.759 | 79.560 |
| **MLP — PyTorch (modelo final)** | **0.843** | **0.628** | **0.845** | **55.910** |
| MLP + threshold ótimo de custo (0.10) | 0.843 | — | — | **24.200** |

> O MLP empata em performance técnica com os melhores baselines, mas
> **reduz o custo de negócio esperado em até 92%** em relação ao baseline
> trivial e em ~67% em relação aos baselines de ML, graças a um recall
> mais alto e à calibração do threshold de decisão por custo.
> Detalhes completos, incluindo análise de viés por subgrupo, estão no
> [Model Card](docs/model_card.md).

## Arquitetura do projeto

```
Dados brutos (CSV)
       │
       ▼
Limpeza + validação de schema (pandera)
       │
       ▼
Pipeline de pré-processamento (ColumnTransformer sklearn)
       │
       ├──► Baselines (Dummy, Logistic Regression, Random Forest)
       │
       └──► MLP (PyTorch) — early stopping, batching, pos_weight
                   │
                   ▼
         Tracking de experimentos (MLflow)
                   │
                   ▼
       Artefatos salvos (models/: pipeline + pesos + metadata)
                   │
                   ▼
            API de inferência (FastAPI)
            /predict · /health
```

## Setup e instalação

### Pré-requisitos

- Python ≥ 3.11
- `pip`

### Instalação do zero

```bash
# 1. Clonar o repositório
git clone <url-do-repositorio>
cd churn-prediction

# 2. Criar e ativar um ambiente virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar o projeto e as dependências de desenvolvimento
pip install -e ".[dev]"

# 4. Baixar o dataset (não versionado no Git — ver .gitignore)
# Baixe o CSV em https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# e salve em: data/raw/Telco-Customer-Churn.csv
```

Todas as dependências e suas versões mínimas estão centralizadas em
[`pyproject.toml`](pyproject.toml) (single source of truth — não há
`requirements.txt` separado).

**Versões usadas no desenvolvimento** (referência, não obrigatórias):
Python 3.12.3 · PyTorch 2.12 · scikit-learn 1.8 · MLflow 3.14 · FastAPI
0.138 · Pydantic 2.13.

## Como usar

Um `Makefile` padroniza todos os comandos do dia a dia:

```bash
make help        # lista todos os comandos disponíveis
make install      # instala o projeto e dependências de dev
make lint         # roda o ruff (linting)
make format       # aplica autofix do ruff
make test         # roda a suíte de testes (pytest)
make test-cov     # roda os testes com relatório de cobertura
make train        # executa o pipeline completo de treino (baselines + MLP)
make run-api      # inicia a API FastAPI localmente (porta 8000)
make mlflow-ui    # abre a interface web do MLflow (porta 5000)
```

### Treinar o modelo do zero

```bash
make train
# equivalente a: python -m churn_prediction.train
```

Isso vai: carregar e limpar os dados → treinar os 3 baselines com CV
estratificada (5-fold) → treinar o MLP com early stopping → avaliar todos
no conjunto de teste hold-out → registrar tudo no MLflow → salvar os
artefatos do MLP em `models/`.

### Rodar a API localmente

```bash
make run-api
# acesse a documentação interativa em http://localhost:8000/docs
```

Exemplo de chamada ao endpoint de predição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
  }'
```

Resposta esperada:

```json
{
  "churn_probability": 0.835,
  "churn_prediction": "Yes",
  "threshold_used": 0.10,
  "risk_tier": "high"
}
```

### Explorar os experimentos no MLflow

```bash
make mlflow-ui
# acesse http://localhost:5000
```

### Explorar a análise exploratória e a modelagem

Os notebooks documentam todo o raciocínio analítico (mas importam a
lógica de produção de `src/`, nunca a duplicam):

- [`notebooks/01_eda_and_baselines.ipynb`](notebooks/01_eda_and_baselines.ipynb) — ML Canvas, EDA completa, baselines.
- [`notebooks/02_mlp_and_comparison.ipynb`](notebooks/02_mlp_and_comparison.ipynb) — arquitetura do MLP, treino, comparação de modelos, trade-off de custo.

## Estrutura do repositório

```
churn-prediction/
├── src/churn_prediction/        # Código de produção (single source of truth)
│   ├── config.py                # Seeds, paths, hiperparâmetros, colunas
│   ├── logging_config.py        # Logging estruturado (JSON)
│   ├── train.py                 # Orquestrador: baselines + MLP + MLflow
│   ├── data/
│   │   ├── loader.py             # Carregamento e limpeza dos dados
│   │   └── schema.py             # Validação de schema (pandera)
│   ├── features/
│   │   └── pipeline.py           # ColumnTransformer (numérico/binário/categórico)
│   ├── models/
│   │   ├── mlp.py                 # Arquitetura da rede neural
│   │   ├── training.py            # Loop de treino, early stopping, seeds
│   │   ├── baselines.py           # Dummy / Logistic Regression / Random Forest
│   │   ├── metrics.py             # Métricas técnicas + custo de negócio
│   │   └── persistence.py         # Salvar/carregar pipeline + modelo
│   └── api/
│       ├── main.py                # App FastAPI (/predict, /health)
│       ├── schemas.py             # Schemas Pydantic de entrada/saída
│       └── middleware.py          # Logging de latência
├── tests/                        # ≥ 3 categorias: schema, smoke, API
│   ├── test_schema.py
│   ├── test_smoke.py
│   └── test_api.py
├── notebooks/                    # EDA e modelagem (documentação analítica)
├── data/raw/                      # Dataset bruto (não versionado, ver .gitignore)
├── models/                        # Artefatos treinados (não versionados, ver .gitignore)
├── docs/
│   ├── model_card.md              # Performance, limitações, vieses
│   ├── deployment_architecture.md # Decisão batch vs. real-time
│   └── monitoring_plan.md         # Métricas, alertas, playbook
├── pyproject.toml                 # Single source of truth (deps, ruff, pytest)
├── Makefile
└── .gitignore
```

## Testes e qualidade

```bash
make test       # 18 testes, cobrindo:
                 #   - test_schema.py: contrato de dados (pandera)
                 #   - test_smoke.py: pipeline, baseline, MLP, métricas
                 #   - test_api.py: /health, /predict, validação Pydantic

make lint        # ruff check — 0 erros
```

**Boas práticas aplicadas:**
- ✅ Seeds fixadas (`RANDOM_SEED=42`) em NumPy, PyTorch e splits do sklearn.
- ✅ Validação cruzada estratificada (5-fold) nos baselines.
- ✅ Early stopping no treino do MLP, com restauração dos melhores pesos.
- ✅ Logging estruturado (JSON) em todo o código — nenhum `print()`.
- ✅ Linting com `ruff` sem erros.
- ✅ Pipeline de pré-processamento idêntico em treino e inferência
  (serializado via `joblib`, sem risco de skew treino/produção).

## Documentação adicional

| Documento | Conteúdo |
|---|---|
| [`docs/model_card.md`](docs/model_card.md) | Performance detalhada, premissas de custo, análise de viés por subgrupo, limitações e cenários de falha. |
| [`docs/deployment_architecture.md`](docs/deployment_architecture.md) | Decisão batch vs. real-time, diagrama de arquitetura, opções de deploy em nuvem. |
| [`docs/monitoring_plan.md`](docs/monitoring_plan.md) | Métricas de monitoramento (operacionais, drift de dados, drift de performance), alertas e playbook de resposta. |

---

## Equipe

Projeto desenvolvido em grupo para o **Tech Challenge — Fase 1**
(Pós-Tech). Vídeo de apresentação (método STAR) disponível em: *(adicionar link)*.
